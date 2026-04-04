import numpy as np

from napari_biopb import ObjectDetectionWidget


def test_test(make_napari_viewer, request):
    """Test widget instantiation."""
    viewer = make_napari_viewer(show=False)
    request.addfinalizer(viewer.close)
    my_widget = ObjectDetectionWidget(viewer)

    assert my_widget
