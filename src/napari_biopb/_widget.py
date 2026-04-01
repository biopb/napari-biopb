import logging
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSpinBox,
    Label,
    LineEdit,
    ProgressBar,
    SpinBox,
    create_widget,
)
from qtpy.QtCore import QTimer
from napari.qt.threading import thread_worker

from ._config import get_grid_params, load_config, save_config

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


class _PersistentComboBox(ComboBox):
    """ComboBox that preserves dynamically set choices during napari state resets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._choices_initialized = True

    def reset_choices(self):
        # Skip reset after initial setup to preserve dynamically populated values
        if not getattr(self, "_choices_initialized", False):
            super().reset_choices()


class _WidgetBase(Container):
    """Base class for biopb napari widgets."""

    def run(self):
        """Run the processing workflow."""
        raise NotImplementedError("Subclass must implement 'run' method.")

    def _snapshot(self) -> dict:
        """Capture current widget settings as a dict."""
        return {w.label: w.value for w in self._elements}

    def _cleanup(self):
        """Reset widget state after processing completes."""
        self._progress_bar.visible = False
        self._run_button.visible = True
        self._run_button.enabled = True
        self._cancel_button.visible = False

        # Disconnect the cancel callback to prevent accumulation
        if self._cancel_callback is not None:
            try:
                self._cancel_button.clicked.disconnect(self._cancel_callback)
            except TypeError:
                pass  # Already disconnected
            self._cancel_callback = None

    def _error(self, exc: Exception):
        """Log error (napari displays it via notification_manager)."""
        self._cleanup()
        logger.error("Processing failed: %s", exc, exc_info=True)

    def _cancel(self, worker):
        """Cancel the running worker.

        Args:
            worker: The thread worker to cancel
        """
        worker.quit()
        self._cancel_button.enabled = False
        # Wait for worker to finish with timeout to prevent indefinite blocking
        worker.await_workers(timeout=5.0)
        self._cleanup()

    def _prepare(self):
        """Prepare widget state before processing starts."""
        self._progress_bar.visible = True
        self._progress_bar.value = 0

        self._run_button.enabled = False
        self._run_button.visible = False

        self._cancel_button.enabled = True
        self._cancel_button.visible = True

        self.out_layer = None

    def _get_data(self, image_layer, is_3d: bool):
        """Extract image data from layer, handling multiscale and RGB.

        Args:
            image_layer: napari Image layer
            is_3d: whether to treat as 3D data

        Returns:
            reshaped image array
        """
        image_data = image_layer.data
        if image_layer.multiscale:
            image_data = image_data[0]
        if image_layer.rgb:
            img_dim = image_data.shape[-4:] if is_3d else image_data.shape[-3:]
            image_data = image_data.reshape((-1,) + img_dim)
        else:
            img_dim = image_data.shape[-3:] if is_3d else image_data.shape[-2:]
            image_data = image_data.reshape((-1,) + img_dim + (1,))

        return image_data

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._cancel_callback = None

        # Load persisted config
        self._config = load_config()

        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )

        self._is3d = create_widget(
            label="3D", annotation=bool, value=self._config.get("3D", False)
        )

        self._server = create_widget(
            value=self._config["server"]["url"],
            label="Server",
            annotation=str,
        )

        self._scheme = ComboBox(
            value=self._config["server"]["scheme"],
            choices=["Auto", "HTTP", "HTTPS"],
            label="Scheme",
        )

        self._progress_bar = ProgressBar(
            label="Running...", value=0, step=1, visible=False
        )

        self._cancel_button = create_widget(
            label="Cancel", widget_type="Button"
        )
        self._cancel_button.visible = False

        self._run_button = create_widget(label="Run", widget_type="Button")
        self._run_button.clicked.connect(self.run)

        self._elements = [
            self._image_layer_combo,
            self._is3d,
            self._server,
            self._scheme,
        ]

    def _save_config(self):
        """Save current widget settings to config file."""
        settings = self._snapshot()
        self._config["server"]["url"] = settings["Server"]
        self._config["server"]["scheme"] = settings["Scheme"]
        self._config["3D"] = settings["3D"]
        save_config(self._config)


class ImageProcessingWidget(_WidgetBase):
    """napari plugin widget for accessing biopb.image.ProcessImage endpoints"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

        logger.debug(
            "Initialized ImageProcessingWidget with server: %s",
            self._server.value,
        )

        # Op selector for choosing which operation to run
        self._op_selector = _PersistentComboBox(
            choices=["<no ops>"],  # Placeholder until populated from server
            label="Op",
            visible=False,  # Hidden until valid schema is loaded
        )

        # Container for dynamically generated kwargs widgets
        self._kwargs_container = Container(label="Kwargs", visible=False)

        # Status label for ops connection status
        self._ops_status = Label(value="Not connected", label="Ops Status")

        # Store schemas for each op (populated from GetOpNames response)
        self._op_schemas = {}

        # Debounce timer for server URL changes (1.5 seconds)
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._fetch_ops)

        # Connect server changes to debounce timer
        self._server.changed.connect(lambda: self._debounce_timer.start(1500))

        # Connect op selector changes to rebuild kwargs widgets
        self._op_selector.changed.connect(self._on_op_changed)

        self.extend(
            self._elements
            + [
                self._ops_status,
                self._op_selector,
                self._kwargs_container,
                self._progress_bar,
                self._cancel_button,
                self._run_button,
            ]
        )

        # Fetch ops during initialization
        self._fetch_ops()

    def _snapshot(self) -> dict:
        """Capture current widget settings including op and kwargs."""
        settings = super()._snapshot()
        settings["Op"] = self._op_selector.value
        for w in self._kwargs_container:
            value = w.value
            if isinstance(w, SpinBox):
                value = int(value)
            settings[w.label] = value
        return settings

    def _fetch_ops(self):
        """Fetch available operations from server asynchronously."""
        from ._grpc import get_op_names

        self._ops_status.value = "Fetching..."
        self._op_selector.visible = False
        self._kwargs_container.visible = False
        settings = self._snapshot()

        def _fetch():
            try:
                ops = get_op_names(settings)
                return ops
            except Exception as e:
                logger.debug("Failed to fetch ops: %s", e)
                return None

        def _on_result(result):
            if result:
                self._op_schemas = dict(result.op_schemas)

                op_list = list(result.names)
                self._op_selector.choices = op_list
                if op_list:
                    self._op_selector.value = op_list[0]

                if len(op_list) == 1:
                    # Single op: hide selector, show op name in status
                    self._ops_status.value = f"Connected: {op_list[0]}"
                    self._op_selector.visible = False
                elif op_list:
                    # Multiple ops: show selector with count in status
                    self._ops_status.value = f"Connected ({len(op_list)} ops)"
                    self._op_selector.visible = True

        self._ops_status.value = "No op schema available"
        self._op_selector.choices = ["<no ops>"]
        self._op_schemas = {}
        self._op_selector.visible = False
        self._kwargs_container.visible = False

        worker = thread_worker(
            _fetch,
            start_thread=True,
            connect={"returned": _on_result},
        )()

    def _on_op_changed(self):
        """Rebuild kwargs widgets when op selection changes."""
        op_name = self._op_selector.value
        self._clear_kwargs_container()

        if not op_name or op_name not in self._op_schemas:
            self._kwargs_container.visible = False
            return

        schema = self._op_schemas[op_name]
        has_kwargs = bool(dict(schema.default_kwargs))
        self._kwargs_container.visible = has_kwargs

        if has_kwargs:
            default_kwargs = dict(schema.default_kwargs)
            logger.debug(
                "Building kwargs widgets for op: %s with kwargs: %s",
                op_name,
                default_kwargs,
            )
            for key, value in default_kwargs.items():
                widget = self._create_widget_for_value(key, value)
                if widget is not None:
                    self._kwargs_container.append(widget)

    def _clear_kwargs_container(self):
        """Clear all widgets from kwargs container."""
        while len(self._kwargs_container) > 0:
            widget = self._kwargs_container[0]
            self._kwargs_container.remove(widget)

    def _create_widget_for_value(self, key: str, value):
        """Create appropriate widget for a kwargs value type.

        Args:
            key: Parameter name (used as label)
            value: Default value (determines widget type)

        Returns:
            magicgui widget or None for unsupported types (list_value)
        """
        # struct_pb2.Value has different fields for different types
        if isinstance(value, bool):
            return CheckBox(value=value, label=key)
        elif isinstance(value, int):
            return SpinBox(value=value, label=key, step=1, format="%d")
        elif isinstance(value, float):
            return FloatSpinBox(value=value, label=key)
        elif isinstance(value, str):
            return LineEdit(value=value, label=key)
        elif isinstance(value, list):
            # TODO: Skip list values for now - complex UI
            logger.debug("Skipping list kwargs: %s", key)
            return None
        else:
            logger.debug("Unknown kwargs type for %s: %s", key, type(value))
            return None

    def run(self):
        from ._grpc import grpc_process_image

        def _update(value):
            if value.shape[-1] == 1:
                value = value.squeeze(-1)
            else:
                value = np.moveaxis(value, -1, 0)

            if self.out_layer is None:
                name = self._image_layer_combo.value.name + "_label"
                _output = np.zeros(
                    [image_data.shape[0], *value.shape], dtype=value.dtype
                )
                if name in self._viewer.layers:
                    self._viewer.layers[name].data = _output
                else:
                    self._viewer.add_image(_output, name=name)

                self.out_layer = self._viewer.layers[name]

            self._progress_bar.increment()
            self.out_layer.data[self.n_results, ...] = value
            self.n_results += 1

            self.out_layer.refresh()

        def _on_success():
            """Callback on successful completion - cleanup and save config."""
            self._cleanup()
            self._save_config()

        self.n_results = 0
        settings = self._snapshot()
        image_layer = settings["Image"]

        with image_layer.dask_optimized_slicing():
            image_data = self._get_data(image_layer, settings["3D"])

            self._prepare()

            self._progress_bar.max = len(image_data)

            worker = grpc_process_image(image_data, settings)

            self._cancel_callback = lambda: self._cancel(worker)
            self._cancel_button.clicked.connect(self._cancel_callback)
            worker.yielded.connect(_update)
            worker.finished.connect(_on_success)
            worker.errored.connect(self._error)

            worker.start()


class ObjectDetectionWidget(_WidgetBase):
    """napari plugin widget for accessing biopb.image.ObjectDetection endpoints"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

        detection_config = self._config.get("detection", {})

        self._threshold = create_widget(
            value=detection_config.get("min_score", 0.4),
            label="Min Score",
            annotation=float,
            widget_type="FloatSlider",
            options={"min": 0, "max": 1},
        )

        self._use_advanced = create_widget(
            value=False,
            label="Advanced",
            annotation=bool,
        )
        self._use_advanced.changed.connect(self._activte_advanced_inputs)

        self._size_hint = create_widget(
            value=detection_config.get("size_hint", 32.0),
            label="Size Hint",
            annotation=float,
            widget_type="FloatSlider",
            options={"min": 10, "max": 200, "visible": False},
        )

        self._nms = ComboBox(
            value=detection_config.get("nms", "Off"),
            choices=["Off", "Iou-0.2", "Iou-0.4", "Iou-0.6", "Iou-0.8"],
            label="NMS",
            visible=False,
        )

        self._aspect_ratio = create_widget(
            value=detection_config.get("z_aspect_ratio", 1.0),
            label="Z Aspect Ratio",
            options={"visible": False},
        )

        self._hidden_elements = [
            self._size_hint,
            self._nms,
            self._aspect_ratio,
        ]

        self._elements += [
            self._threshold,
            self._use_advanced,
        ]
        self._elements += self._hidden_elements

        # append into/extend the container with your widgets
        self.extend(
            self._elements
            + [
                self._progress_bar,
                self._cancel_button,
                self._run_button,
            ]
        )

    def _activte_advanced_inputs(self):
        """Toggle visibility of advanced settings widgets."""
        for ctl in self._hidden_elements:
            ctl.visible = self._use_advanced.value

    def _get_grid_positions(
        self, image, settings: dict
    ) -> List[Tuple[slice, ...]]:
        """Compute grid positions for tiled processing.

        Args:
            image: input image array
            settings: widget settings dict

        Returns:
            list of slice tuples defining patch positions
        """
        is_3d = settings["3D"]
        grid_size, stride = get_grid_params(is_3d, self._config)

        pos_pars = (
            image.shape[:-1],
            grid_size,
            stride,
        )

        grid_start = [
            slice(0, max(d - (gs - ss), 1), ss) for d, gs, ss in zip(*pos_pars)
        ]
        grid_start = np.moveaxis(np.mgrid[grid_start], 0, -1)
        grid_start = grid_start.reshape(-1, image.ndim - 1)

        grids = []
        for x in grid_start:
            slc = (slice(x[i], x[i] + gs) for i, gs in enumerate(pos_pars[1]))
            grids.append(tuple(slc))

        return grids

    def _save_config(self):
        """Save current widget settings to config file."""
        settings = self._snapshot()
        self._config["server"]["url"] = settings["Server"]
        self._config["server"]["scheme"] = settings["Scheme"]
        self._config["3D"] = settings["3D"]
        self._config["detection"]["min_score"] = settings["Min Score"]
        self._config["detection"]["size_hint"] = settings["Size Hint"]
        self._config["detection"]["nms"] = settings["NMS"]
        self._config["detection"]["z_aspect_ratio"] = settings[
            "Z Aspect Ratio"
        ]
        save_config(self._config)

    def run(self):
        from ._grpc import grpc_object_detection

        self.n_results = 0

        settings = self._snapshot()
        image_layer = settings["Image"]

        def _update(value):
            if value is None:  # patch prediction
                self._progress_bar.increment()
            else:  # full image prediction
                if self.out_layer is None:
                    name = self._image_layer_combo.value.name + "_label"
                    _output = np.zeros(image_data.shape[:-1], dtype=int)
                    if name in self._viewer.layers:
                        self._viewer.layers[name].data = _output
                    else:
                        self._viewer.add_labels(_output, name=name)

                    self.out_layer = self._viewer.layers[name]

                self.out_layer.data[self.n_results, ...] = value
                self.n_results += 1
                self.out_layer.refresh()

        def _on_success():
            """Callback on successful completion - cleanup and save config."""
            self._cleanup()
            self._save_config()

        with image_layer.dask_optimized_slicing():
            image_data = self._get_data(image_layer, settings["3D"])

            grid_positions = self._get_grid_positions(image_data[0], settings)

            self._progress_bar.max = len(image_data) * len(grid_positions)

            self._prepare()

            worker = grpc_object_detection(
                image_data, settings, grid_positions
            )

            self._cancel_callback = lambda: self._cancel(worker)
            self._cancel_button.clicked.connect(self._cancel_callback)
            worker.yielded.connect(_update)
            worker.finished.connect(_on_success)
            worker.errored.connect(self._error)

            worker.start()
