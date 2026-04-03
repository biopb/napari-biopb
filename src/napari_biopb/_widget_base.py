import logging
from typing import TYPE_CHECKING

from magicgui.widgets import ComboBox, Container, ProgressBar, create_widget
from napari.qt.threading import thread_worker

from ._config import load_config, save_config

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
        ]

    def _save_config(self):
        """Save current widget settings to config file."""
        settings = self._snapshot()
        self._config["server"]["url"] = settings["Server"]
        self._config["3D"] = settings["3D"]
        save_config(self._config)
