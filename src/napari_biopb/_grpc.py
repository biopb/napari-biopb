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

    # image = np.ascontiguousarray(image, ">f2")

    print(image.shape)
    dt_str = image.dtype.str

    image_data = proto.ImageData(
        pixels=proto.Pixels(
            bindata=proto.BinData(
                data=image.tobytes(),
                endianness=1 if dt_str[0] == "<" else 0,
            ),
            size_c=image.shape[-1],
            size_x=image.shape[-2],
            size_y=image.shape[-3],
            size_z=image.shape[-4],
            dimension_order="CXYZT",
            dtype=dt_str,
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


def _get_channel(values: dict):
    server_url = values["Server"]
    if ":" in server_url:
        _, port = server_url.split(":")
    else:
        server_url += ":443"
        port = 443

    if port == 443:
        return grpc.secure_channel(
            target=server_url,
            credentials=grpc.ssl_channel_credentials(),
            options=[("grpc.max_receive_message_length", 1024 * 1024 * 512)],
        )
    else:
        return grpc.insecure_channel(
            target=server_url,
            options=[("grpc.max_receive_message_length", 1024 * 1024 * 512)],
        )


def _get_settings(values: dict):
    nms_iou = values["NMS Iou"] if values["NMS"] else 0

    return proto.DetectionSettings(
        min_score=values["Min Score"],
        nms_iou=nms_iou,
        cell_diameter_hint=values["Size Hint"],
    )


def _render_meshes(response, label):
    from vedo import Mesh

    meshes = []
    for det in response.detections:
        verts, cells = [], []
        for vert in det.roi.mesh.verts:
            verts.append(
                [
                    vert.z,
                    vert.y,
                    vert.x,
                ]
            )
        for face in det.roi.mesh.faces:
            cells.append([face.p1, face.p2, face.p3])
        meshes.append(Mesh([verts, cells]))

    for k, mesh in reversed(list(enumerate(meshes))):
        origin = np.floor(mesh.bounds()[::2]).astype(int)
        origin = np.maximum(origin, 0)
        max_size = np.array(label.shape) - origin
        vol = mesh.binarize(
            values=(k + 1, 0),
            spacing=[1, 1, 1],
            origin=origin + 0.5,
        )
        vol_d = vol.tonumpy()[: max_size[0], : max_size[1], : max_size[2]]
        size = tuple(vol_d.shape)
        region = label[
            origin[0] : origin[0] + size[0],
            origin[1] : origin[1] + size[1],
            origin[2] : origin[2] + size[2],
        ]
        region[...] = np.maximum(region, vol_d)

    return label


def _generate_label(response, label):
    if label.ndim == 2:
        for k, det in enumerate(response.detections):
            polygon = [[p.x, p.y] for p in det.roi.polygon.points]
            polygon = np.round(np.array(polygon)).astype(int)

            cv2.fillPoly(label, [polygon], k + 1)
    elif label.ndim == 3:
        _render_meshes(response, label)
    else:
        raise ValueError(
            f"supplied label template is not 2d or 3d: {label.shape}"
        )

    return label


def grpc_call(widget: BiopbImageWidget) -> np.ndarray:
    """make grpc call based on current widget values"""
    widget_values = widget.snapshot()
    progress_bar = widget._progress_bar

    image_layer = widget_values["Image"]
    image_data = image_layer.data
    is3d = widget_values["3D"]

    # proprocess
    if image_layer.rgb:
        img_dim = image_data.shape[-4:] if is3d else image_data.shape[-3:]
        image_data = image_data.reshape((-1,) + img_dim)
    else:
        img_dim = image_data.shape[-3:] if is3d else image_data.shape[-2:]
        image_data = image_data.reshape((-1,) + img_dim + (1,))

    assert image_data.ndim == 4 or image_data.ndim == 5
    progress_bar.max = len(image_data)

    settings = _get_settings(widget_values)

    # call server
    with _get_channel(widget_values) as channel:
        stub = proto.ObjectDetectionStub(channel)

        labels = []
        for image in image_data:
            request = _build_request(image, settings)

            timeout = 300 if is3d else 5

            response = stub.RunDetection(request, timeout=timeout)

            print(f"Detected {len(response.detections)} cells")

            labels.append(
                _generate_label(
                    response, np.zeros(image_data.shape[1:-1], dtype="uint16")
                )
            )
            progress_bar.increment()

    if image_layer.rgb:
        labels = np.reshape(labels, image_layer.data.shape[:-1])
    else:
        labels = np.reshape(labels, image_layer.data.shape)

    return labels
