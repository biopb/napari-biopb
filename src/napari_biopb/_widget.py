from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import ComboBox, Container, ProgressBar, create_widget

if TYPE_CHECKING:
    import napari


class BiopbImageWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )

        # self._roi = create_widget(
        #     label="ROI",
        #     annotation="napari.layers.Shapes",
        #     options={"nullable":True},
        # )

        self._is3d = create_widget(label="3D", annotation=bool)

        self._server = create_widget(
            value="lacss.biopb.org",
            label="Server",
            annotation=str,
        )

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

        self._grid_size = ComboBox(
            value=1024,
            choices=[512, 768, 1024, 2048],
            label="Grid Size",
            visible=False,
        )

        self._grid_size_3d = ComboBox(
            value=256,
            choices=[256, 384, 512],
            label="Grid Size 3D",
            visible=False,
        )

        self._scheme = ComboBox(
            value="Auto",
            choices=["Auto", "HTTP", "HTTPS"],
            label="Scheme",
            visible=False,
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
            # self._roi,
            self._is3d,
            self._server,
            self._threshold,
            self._use_advanced,
            self._size_hint,
            self._nms,
            self._aspect_ratio,
            self._grid_size,
            self._grid_size_3d,
            self._scheme,
            self._progress_bar,
            self._cancel_button,
            self._run_button,
        ]

        # append into/extend the container with your widgets
        self.extend(self._elements)

    def _activte_advanced_inputs(self):
        for ctl in [
            self._size_hint,
            self._nms,
            self._aspect_ratio,
            self._grid_size,
            self._grid_size_3d,
            self._scheme,
        ]:
            ctl.visible = self._use_advanced.value

    def _get_grid_positions(self, image, settings):
        if settings["3D"]:
            gs_ = int(settings["Grid Size 3D"])
            ss_ = max(gs_ - int(settings["Size Hint"] * 2), 192)
            ratio = settings["Z Aspect Ratio"]
            pos_pars = (
                image.shape[:-1],
                (int(gs_ / ratio), gs_, gs_),
                (int(ss_ / ratio), ss_, ss_),
            )
        else:
            gs_ = int(settings["Grid Size"])
            ss_ = gs_ - int(settings["Size Hint"] * 2)
            pos_pars = (
                image.shape[:-1],
                (gs_, gs_),
                (ss_, ss_),
            )

        grid_start = [slice(0, max(d-(gs-ss), 1), ss) for d, gs, ss in zip(*pos_pars)]
        grid_start = np.moveaxis(np.mgrid[grid_start], 0, -1)
        grid_start = grid_start.reshape(-1, image.ndim - 1)

        grids = []
        for x in grid_start:
            slc = (slice(x[i], x[i] + gs) for i, gs in enumerate(pos_pars[1]))
            grids.append(tuple(slc))

        return grids


    def _snapshot(self):
        return {w.label: w.value for w in self._elements}


    def run(self):
        from ._grpc import grpc_call

        name = self._image_layer_combo.value.name + "_label"

        settings = self._snapshot()
        image_layer = settings["Image"]
        image_data = image_layer.data

        if image_layer.rgb:
            img_dim = image_data.shape[-4:] if settings["3D"] else image_data.shape[-3:]
            image_data = image_data.reshape((-1,) + img_dim)
        else:
            img_dim = image_data.shape[-3:] if settings["3D"] else image_data.shape[-2:]
            image_data = image_data.reshape((-1,) + img_dim + (1,))

        grid_positions = self._get_grid_positions(image_data[0], settings)
        self._progress_bar.max = len(image_data) * len(grid_positions)

        labels = np.zeros_like(image_data, dtype="int16")
        self.n_results = 0

        def _update(value):
            if value is None: # patch prediction
                self._progress_bar.increment()

            else: # full image prediction
                labels[self.n_results, ...] = value
                self.n_results += 1

                if name in self._viewer.layers:
                    self._viewer.layers[name].data = labels
                else:
                    self._viewer.add_labels(labels, name=name)

        def _cleanup():
            self._progress_bar.visible = False
            self._run_button.visible = True
            self._run_button.enabled = True
            self._cancel_button.visible = False

        def _error(exc):
            _cleanup()
            raise exc

        def _cancel():
            worker.quit()
            _cleanup()

        self._progress_bar.visible = True
        self._progress_bar.value = 0

        self._run_button.enabled = False
        self._run_button.visible = False

        self._cancel_button.visible = True
        self._cancel_button.clicked.connect(_cancel)

        worker = grpc_call(image_data, settings, grid_positions)

        worker.yielded.connect(_update)
        worker.finished.connect(_cleanup)
        worker.errored.connect(_error)

        worker.start()
