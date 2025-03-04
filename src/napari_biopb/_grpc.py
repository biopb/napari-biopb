from typing import Generator

import biopb.image as proto
import cv2
import grpc
import numpy as np

from napari.qt.threading import thread_worker


def _build_request(image: np.ndarray, values: dict) -> proto.DetectionRequest:
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
            physical_size_x=values["Pixel Size X"],
            physical_size_y=values["Pixel Size Y"],
            physical_size_z=values["Pixel Size Z"],
            dimension_order="CXYZT",
            dtype=dt_str,
        ),
    )

    request = proto.DetectionRequest(
        image_data=image_data,
        detection_settings=_get_settings(values),
    )

    return request


def _get_channel(values: dict):
    server_url = values["Server"]
    if ":" in server_url:
        _, port = server_url.split(":")
    else:
        server_url += ":443"
        port = 443

    scheme = values["Scheme"]
    if scheme == "Auto":
        scheme = "HTTPS" if port == 443 else "HTTP"
    if scheme == "HTTPS":
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
    nms_values = {
        "Off": 0.0,
        "Iou-0.2": 0.2,
        "Iou-0.4": 0.4,
        "Iou-0.6": 0.6,
        "Iou-0.8": 0.8,
    }
    nms_iou = nms_values[values["NMS"]]

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

    color = 1
    for mesh in meshes[::-1]:
        origin = np.floor(mesh.bounds()[::2]).astype(int)
        origin = np.maximum(origin, 0)

        max_size = np.array(label.shape) - origin

        vol = mesh.binarize(
            values=(color, 0),
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

        color = color + 1

    return label


def _generate_label(response, label)-> np.ndarray:
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


@thread_worker
def grpc_call(image_data: np.ndarray, settings: dict) -> Generator[np.ndarray, None, None]:
    is3d = settings["3D"]
    if is3d:
        assert image_data.ndim == 5
    else:
        assert image_data.ndim == 5

    # call server
    with _get_channel(settings) as channel:
        stub = proto.ObjectDetectionStub(channel)

        for image in image_data:
            request = _build_request(image, settings)

            timeout = 300 if is3d else 5

            response = stub.RunDetection(request, timeout=timeout)

            print(f"Detected {len(response.detections)} cells")

            yield _generate_label(
                response, np.zeros(image_data.shape[1:-1], dtype="uint16")
            )

