# def test_image_threshold_widget(make_napari_viewer):
#     viewer = make_napari_viewer()
#     layer = viewer.add_image(np.random.random((100, 100)))
#     my_widget = BiopbImageWidget(viewer)

# my_widget._image_layer_combo.value = layer
# my_widget._threshold_slider.value = 0.5

# my_widget.run()
# assert len(viewer.layers) == 2
