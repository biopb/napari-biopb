import numpy as np

from napari_biopb import ObjectDetectionWidget


def test_test(make_napari_viewer, request):
    """Test widget instantiation."""
    viewer = make_napari_viewer()
    request.addfinalizer(viewer.close)
    layer = viewer.add_image(np.random.random((100, 100)))
    my_widget = ObjectDetectionWidget(viewer)

    assert my_widget
