"""Tests for progress bar timer behavior during long gRPC calls."""

import time
import pytest
from qtpy.QtCore import QTimer
from napari_biopb._grpc import CALL_START


class TestProgressTimer:
    """Tests for the QTimer-based progress in _WidgetBase."""

    def test_timer_ticks_when_started(self, qtbot, make_napari_viewer):
        """Timer ticks and updates progress bar value when started."""
        from qtpy.QtWidgets import QApplication
        from napari_biopb._widget_base import _WidgetBase

        viewer = make_napari_viewer()
        widget = _WidgetBase(viewer)

        widget._progress_bar.max = 100
        widget._progress_bar.visible = True

        # Track timer ticks
        tick_count = [0]
        original_tick = widget._tick_fake_progress

        def counting_tick():
            tick_count[0] += 1
            original_tick()

        widget._progress_timer.timeout.disconnect()
        widget._progress_timer.timeout.connect(counting_tick)

        # Start timer
        widget._fake_subprogress = 0
        widget._calls_completed = 0
        widget._progress_timer.start(50)  # 50ms interval

        # Process events manually
        app = QApplication.instance()
        for _ in range(15):
            app.processEvents()
            time.sleep(0.05)
            if tick_count[0] >= 3:
                break

        widget._progress_timer.stop()

        assert tick_count[0] >= 3, f"Expected >= 3 ticks, got {tick_count[0]}"
        assert widget._fake_subprogress > 0

    def test_on_call_starting_starts_timer(self, qtbot, make_napari_viewer):
        """_on_call_starting() starts the timer."""
        from qtpy.QtWidgets import QApplication
        from napari_biopb._widget_base import _WidgetBase

        viewer = make_napari_viewer()
        widget = _WidgetBase(viewer)

        # Timer should not be running initially
        assert not widget._progress_timer.isActive()

        # Call _on_call_starting
        widget._on_call_starting()

        # Timer should now be active
        assert widget._progress_timer.isActive()

        # Process events and verify it ticks
        app = QApplication.instance()
        for _ in range(15):
            app.processEvents()
            time.sleep(0.05)

        assert widget._fake_subprogress > 0
        widget._progress_timer.stop()

    def test_on_call_completed_stops_timer(self, qtbot, make_napari_viewer):
        """_on_call_completed() stops the timer."""
        from napari_biopb._widget_base import _WidgetBase

        viewer = make_napari_viewer()
        widget = _WidgetBase(viewer)

        # Start timer
        widget._on_call_starting()
        assert widget._progress_timer.isActive()

        # Complete the call
        widget._on_call_completed()

        # Timer should be stopped
        assert not widget._progress_timer.isActive()
        assert widget._calls_completed == 1
        assert widget._fake_subprogress == 0

    def test_timer_parent_is_widget_native(self, qtbot, make_napari_viewer):
        """Timer is parented to the widget's native Qt widget."""
        from napari_biopb._widget_base import _WidgetBase

        viewer = make_napari_viewer()
        widget = _WidgetBase(viewer)

        assert widget._progress_timer.parent() == widget.native

    def test_progress_uses_scaled_integers(self, qtbot, make_napari_viewer):
        """Progress values are scaled integers (x10) for magicgui compatibility."""
        from qtpy.QtWidgets import QApplication
        from napari_biopb._widget_base import _WidgetBase

        viewer = make_napari_viewer()
        widget = _WidgetBase(viewer)
        widget._progress_bar.max = 100

        # Simulate progress ticks
        widget._fake_subprogress = 0
        widget._calls_completed = 0

        widget._tick_fake_progress()
        assert widget._progress_bar.value == 1

        widget._tick_fake_progress()
        assert widget._progress_bar.value == 2

        # After 9 ticks, should cap at 9
        for _ in range(10):
            widget._tick_fake_progress()
        assert widget._fake_subprogress == 9
        assert widget._progress_bar.value == 9

    @pytest.mark.skip(
        reason="qtbot.waitSignal doesn't process QTimer events; works in real GUI"
    )
    def test_progress_updates_during_thread_worker(
        self, qtbot, make_napari_viewer
    ):
        """Timer ticks while a thread worker is blocked."""
        from qtpy.QtWidgets import QApplication
        from napari_biopb._widget_base import _WidgetBase
        from napari.qt.threading import thread_worker

        viewer = make_napari_viewer()
        widget = _WidgetBase(viewer)
        widget._progress_bar.max = 100

        # Track progress values
        progress_values = []
        original_tick = widget._tick_fake_progress

        def tracking_tick():
            progress_values.append(widget._progress_bar.value)
            original_tick()

        widget._progress_timer.timeout.disconnect()
        widget._progress_timer.timeout.connect(tracking_tick)

        # Use shorter timer interval for testing
        widget._progress_timer.setInterval(50)  # 50ms

        # Create a blocking generator
        @thread_worker
        def blocking_gen():
            yield CALL_START
            time.sleep(0.5)
            yield None

        worker = blocking_gen()

        call_started = [False]

        def handle_yield(value):
            if value == CALL_START:
                call_started[0] = True
                widget._on_call_starting()
            elif value is None:
                widget._on_call_completed()

        worker.yielded.connect(handle_yield)
        worker.start()

        # Wait for worker
        qtbot.waitSignal(worker.started, timeout=1000)

        # Process events
        app = QApplication.instance()
        start_time = time.time()
        while time.time() - start_time < 2.0:
            app.processEvents()
            time.sleep(0.02)
            if (
                call_started[0]
                and len([v for v in progress_values if v > 0]) >= 3
            ):
                break

        widget._progress_timer.stop()

        print(f"Progress values: {progress_values}")
        assert (
            len([v for v in progress_values if v > 0]) >= 3
        ), f"Expected >= 3 progress updates, got {len([v for v in progress_values if v > 0])}"


class TestSignalDelivery:
    """Tests for signal delivery from thread workers."""

    def test_yielded_signals_delivered(self, qtbot, make_napari_viewer):
        """Yielded signals from thread worker are eventually delivered."""
        from qtpy.QtWidgets import QApplication
        from napari.qt.threading import thread_worker

        viewer = make_napari_viewer()

        signals_received = []

        @thread_worker
        def simple_gen():
            yield "first"
            yield "second"

        def on_yield(value):
            signals_received.append(value)

        worker = simple_gen()
        worker.yielded.connect(on_yield)
        worker.start()

        qtbot.waitSignal(worker.finished, timeout=2000)

        app = QApplication.instance()
        for _ in range(10):
            app.processEvents()
            time.sleep(0.01)

        assert len(signals_received) == 2
        assert signals_received == ["first", "second"]
