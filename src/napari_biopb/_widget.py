from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import ComboBox, Container, ProgressBar, create_widget

if TYPE_CHECKING:
    import napari


class _WidgetBase(Container):
    def run(self):
        raise NotImplementedError("Subclass must implement 'run' method.")

    def _snapshot(self):
        return {w.label: w.value for w in self._elements}

    def _cleanup(self):
        self._progress_bar.visible = False
        self._run_button.visible = True
        self._run_button.enabled = True
        self._cancel_button.visible = False

    def _error(self, exc):
        self._cleanup()
        raise exc

    def _cancel(self, worker):
        worker.quit()
        self._cancel_button.enabled = False
        worker.await_workers()
        self._cleanup()

    def _prepare(self):
        self._progress_bar.visible = True
        self._progress_bar.value = 0

        self._run_button.enabled = False
        self._run_button.visible = False

        self._cancel_button.enabled = True
        self._cancel_button.visible = True

        self.out_layer = None

    def _get_data(self, image_layer, is_3d):
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

        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )

        self._is3d = create_widget(label="3D", annotation=bool)

        self._server = create_widget(
            value="lacss.biopb.org",
            label="Server",
            annotation=str,
        )

        self._scheme = ComboBox(
            value="Auto",
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


class ImageProcessingWidget(_WidgetBase):
    """napari plugin widget for accessing biopb.image.ProcessImage endpoints"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.extend(
            self._elements
            + [
                self._progress_bar,
                self._cancel_button,
                self._run_button,
            ]
        )

    def run(self):
        from ._grpc import grpc_process_image

        def _update(value):
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

        self.n_results = 0
        settings = self._snapshot()
        image_layer = settings["Image"]

        with image_layer.dask_optimized_slicing():
            image_data = self._get_data(image_layer, settings["3D"])

            self._prepare()

            self._progress_bar.max = len(image_data)

            worker = grpc_process_image(image_data, settings)

            self._cancel_button.clicked.connect(lambda: self._cancel(worker))
            worker.yielded.connect(_update)
            worker.finished.connect(self._cleanup)
            worker.errored.connect(self._error)

            worker.start()


class ObjectDetectionWidget(_WidgetBase):
    """napari plugin widget for accessing biopb.image.ObjectDetection endpoints"""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self._threshold = create_widget(
            value=0.4,
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
            value=32.0,
            label="Size Hint",
            annotation=float,
            widget_type="FloatSlider",
            options={"min": 10, "max": 200, "visible": False},
        )

        self._nms = ComboBox(
            value="Off",
            choices=["Off", "Iou-0.2", "Iou-0.4", "Iou-0.6", "Iou-0.8"],
            label="NMS",
            visible=False,
        )

        self._aspect_ratio = create_widget(
            value=1.0,
            label="Z Aspect Ratio",
            options={"visible": False},
        )

        self._grid_size_limit = create_widget(
            value=64,
            label="Grid Size Limit [MPixel]",
            options={"visible": False},
        )

        self._hidden_elements = [
            self._size_hint,
            self._nms,
            self._aspect_ratio,
            self._grid_size_limit,
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
        for ctl in self._hidden_elements:
            ctl.visible = self._use_advanced.value

    def _get_grid_positions(self, image, settings):
        if settings["3D"]:
            pos_pars = (
                image.shape[:-1],
                np.array((64, 512, 512), dtype=int),
                np.array((48, 480, 480), dtype=int),
            )
        else:
            # gs_ = int(settings["Grid Size Limit [MPixel]"] ** 0.5) * 1024
            # ss_ = gs_ - int(settings["Size Hint"] * 4)
            pos_pars = (
                image.shape[:-1],
                np.array((4096, 4096), dtype=int),
                np.array((4000, 4000), dtype=int),
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

        with image_layer.dask_optimized_slicing():
            image_data = self._get_data(image_layer, settings["3D"])

            grid_positions = self._get_grid_positions(image_data[0], settings)

            self._progress_bar.max = len(image_data) * len(grid_positions)

            self._prepare()

            worker = grpc_object_detection(
                image_data, settings, grid_positions
            )

            self._cancel_button.clicked.connect(lambda: self._cancel(worker))
            worker.yielded.connect(_update)
            worker.finished.connect(self._cleanup)
            worker.errored.connect(self._error)

            worker.start()
