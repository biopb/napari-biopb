import biopb.image as proto
import cv2
import grpc
import numpy as np

from ._widget import BiopbImageWidget


def _build_request(
    image: np.ndarray, settings: proto.DetectionSettings | None = None
) -> proto.DetectionRequest:
    """Serialize a np image array as ImageData protobuf"""
    assert (
        image.ndim == 3 or image.ndim == 4
    ), f"image received is neither 2D nor 3D, shape={image.shape}."

    if image.ndim == 3:
        image = image[None, ...]

    image = np.ascontiguousarray(image, ">f2")

    image_data = proto.ImageData(
        pixels=proto.Pixels(
            bindata=proto.BinData(data=image.tobytes()),
            size_c=image.shape[-1],
            size_x=image.shape[-2],
            size_y=image.shape[-3],
            size_z=image.shape[-4],
            dimension_order="CXYZT",
            dtype="f2",
        ),
    )

    if settings is not None:
        request = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=settings,
        )
    else:
        request = proto.DetectionRequest(
            image_data=image_data,
        )

    return request


def _get_channel(widget: BiopbImageWidget):
    server_url = widget._server.value
    if ":" in server_url:
        _, port = server_url.split(":")
    else:
        server_url += ":443"
        port = 443

    if port == 443:
        return grpc.secure_channel(
            target=server_url, credentials=grpc.ssl_channel_credentials()
        )
    else:
        return grpc.insecure_channel(
            target=server_url,
        )


def _get_settings(widget: BiopbImageWidget):
    nms_iou = widget._nms_iou.value if widget._nms.value else 0

    return proto.DetectionSettings(
        min_score=widget._threshold.value,
        nms_iou=nms_iou,
        cell_diameter_hint=widget._size_hint.value,
    )


def _generate_label(response, label):
    for k, det in enumerate(response.detections):
        polygon = [[p.x, p.y] for p in det.roi.polygon.points]
        polygon = np.round(np.array(polygon)).astype(int)

        cv2.fillPoly(label, [polygon], k + 1)

    return label


def grpc_call(widget: BiopbImageWidget) -> np.ndarray:
    """make grpc call based on current widget values"""
    image_layer = widget._image_layer_combo.value
    image_data = image_layer.data

    # proprocess
    if image_layer.rgb:
        img_dim = image_data.shape[-3:]
        image_data = image_data.reshape((-1,) + img_dim)
    else:
        img_dim = image_data.shape[-2:]
        image_data = image_data.reshape((-1,) + img_dim + (1,))

    assert image_data.ndim == 4
    widget._progress_bar.max = len(image_data)

    settings = _get_settings(widget)

    # call server
    with _get_channel(widget) as channel:
        stub = proto.ObjectDetectionStub(channel)

        labels = []
        for image in image_data:
            request = _build_request(image, settings)
            response = stub.RunDetection(request)
            labels.append(
                _generate_label(
                    response, np.zeros(image_data.shape[1:-1], dtype="uint16")
                )
            )
            widget._progress_bar.increment()

    if image_layer.rgb:
        labels = np.reshape(labels, image_layer.data.shape[:-1])
    else:
        labels = np.reshape(labels, image_layer.data.shape)

    return labels
