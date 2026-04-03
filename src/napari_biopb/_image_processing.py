import logging
from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import (
    CheckBox,
    Container,
    FloatSpinBox,
    Label,
    LineEdit,
    SpinBox,
)
from qtpy.QtCore import QTimer
from napari.qt.threading import thread_worker

from ._chunking import (
    ResultBuilder,
    _get_axis_mapping,
    _get_iter_spec,
    _validate_data_shape,
)
from ._widget_base import _PersistentComboBox, _WidgetBase

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


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

        settings = self._snapshot()
        image_layer = settings["Image"]
        op_name = settings["Op"]

        # Get axis order from raw layer data (respects napari heuristics)
        axis_order = _get_axis_mapping(image_layer, settings["3D"])

        # Get hint from op schema
        hint = None
        if op_name in self._op_schemas:
            hint = self._op_schemas[op_name].input_shape_hint

        # Handle multiscale: use highest resolution level
        image_data = image_layer.data
        if image_layer.multiscale:
            image_data = image_data[0]

        # Compute iteration spec
        iter_spec = _get_iter_spec(axis_order, hint)

        # Validate
        _validate_data_shape(image_data, axis_order, hint)

        # Prepare result builder and progress
        result_builder = ResultBuilder(image_data.shape, iter_spec.iter_dims)
        n_iterations = (
            1
            if not iter_spec.iter_dims
            else int(
                np.prod([image_data.shape[d] for d in iter_spec.iter_dims])
            )
        )

        self._prepare()
        self._progress_bar.max = n_iterations

        def _update(value):
            result_chunk, position, squeezed_dims = value

            result_builder.add_result(position, result_chunk, squeezed_dims)
            self._progress_bar.increment()

            if result_builder.buffer is not None:
                if self.out_layer is None:
                    name = image_layer.name + "_processed"
                    if name in self._viewer.layers:
                        self._viewer.layers[name].data = result_builder.buffer
                    else:
                        self._viewer.add_image(
                            result_builder.buffer,
                            name=name,
                        )
                    self.out_layer = self._viewer.layers[name]
                    self.out_layer.reset_contrast_limits()
                else:
                    self.out_layer.data = result_builder.buffer
                    self.out_layer.refresh()

        def _on_success():
            """Callback on successful completion - cleanup and save config."""
            if self.out_layer is not None:
                self.out_layer.reset_contrast_limits()
            self._cleanup()
            self._save_config()

        # Use dask_optimized_slicing for large images
        with image_layer.dask_optimized_slicing():
            worker = grpc_process_image(image_data, settings, iter_spec)

        self._cancel_callback = lambda: self._cancel(worker)
        self._cancel_button.clicked.connect(self._cancel_callback)
        worker.yielded.connect(_update)
        worker.finished.connect(_on_success)
        worker.errored.connect(self._error)

        worker.start()
