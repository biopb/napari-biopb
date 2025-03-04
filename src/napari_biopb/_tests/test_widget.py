import numpy as np
from napari_biopb import BiopbImageWidget

def test_test(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))
    my_widget = BiopbImageWidget(viewer)
    assert True
