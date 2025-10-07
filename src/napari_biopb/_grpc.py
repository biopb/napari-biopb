import logging
from typing import Generator

import biopb.image as proto
import cv2
import grpc
import numpy as np
from biopb.image.utils import deserialize_to_numpy, serialize_from_numpy
from napari.qt.threading import thread_worker

from ._typing import napari_data

logger = logging.getLogger(__name__)


def _box_intersection(boxes_a, boxes_b):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxes_a: [..., N, 2d]
      boxes_b: [..., M, 2d]

    Returns:
      a float Tensor with shape [..., N, M] representing pairwise intersections.
    """
    from numpy import maximum, minimum

    ndim = boxes_a.shape[-1] // 2
    assert ndim * 2 == boxes_a.shape[-1]
    assert ndim * 2 == boxes_b.shape[-1]

    min_vals_1 = boxes_a[..., None, :ndim]  # [..., N, 1, d]
    max_vals_1 = boxes_a[..., None, ndim:]
    min_vals_2 = boxes_b[..., None, :, :ndim]  # [..., 1, M, d]
    max_vals_2 = boxes_b[..., None, :, ndim:]

    min_max = minimum(max_vals_1, max_vals_2)  # [..., N, M, d]
    max_min = maximum(min_vals_1, min_vals_2)

    intersects = maximum(0, min_max - max_min)  # [..., N, M, d]

    return intersects.prod(axis=-1)


def _filter_boxes(boxes, threshold=0.75):
    """Filter boxes based on the overlap. For each box, check if it is mostly enclosed by another box. If so, remove it.

    Args:
      boxes: [N, 4/6]
      threshold: float

    Returns:
      a boolean tensor with shape [N]
    """
    areas = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    intersections = _box_intersection(boxes, boxes)  # [..., N, N]

    its_area_ratio = intersections / (areas[..., None] + 1e-6)
    np.fill_diagonal(its_area_ratio, 0)

    # scan from the lowest score
    bm = np.ones([its_area_ratio.shape[-1]], dtype=bool)
    for i in range(its_area_ratio.shape[-1] - 1, -1, -1):
        if np.any(its_area_ratio[i] > threshold):
            bm[i] = False
            its_area_ratio[..., i] = 0

    return bm


def _encode_image(image: np.ndarray, z_ratio=1.0):
    assert (
        image.ndim == 3 or image.ndim == 4
    ), f"image received is neither 2D nor 3D, shape={image.shape}."

    if image.ndim == 3:
        image = image[None, ...]

    pixels = serialize_from_numpy(
        image,
        physical_size_x=1.0,
        physical_size_y=1.0,
        physical_size_z=z_ratio,
    )

    return pixels


def _object_detection_build_request(
    image: np.ndarray, settings: dict
) -> proto.DetectionRequest:
    """Serialize a np image array as ImageData protobuf"""
    pixels = _encode_image(image, settings["Z Aspect Ratio"])

    request = proto.DetectionRequest(
        image_data=proto.ImageData(pixels=pixels),
        detection_settings=_get_detection_settings(settings),
    )

    return request


def _get_grpc_channel(settings: dict):
    server_url = settings["Server"]
    if ":" in server_url:
        _, port = server_url.split(":")
    else:
        server_url += ":443"
        port = 443

    scheme = settings["Scheme"]
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


def _get_detection_settings(settings: dict):
    nms_values = {
        "Off": 0.0,
        "Iou-0.2": 0.2,
        "Iou-0.4": 0.4,
        "Iou-0.6": 0.6,
        "Iou-0.8": 0.8,
    }
    nms_iou = nms_values[settings["NMS"]]

    return proto.DetectionSettings(
        min_score=settings["Min Score"],
        nms_iou=nms_iou,
        cell_diameter_hint=settings["Size Hint"],
    )


def _render_meshes(response, label, post_process=False):
    from vedo import Mesh

    if post_process:
        bboxes = []
        for det in response.detections:
            x = [v.x for v in det.roi.mesh.verts]
            y = [v.y for v in det.roi.mesh.verts]
            z = [v.z for v in det.roi.mesh.verts]
            bboxes.append([min(z), min(y), min(x), max(z), max(y), max(z)])

        bm = _filter_boxes(np.array(bboxes))

    else:
        bm = [True] * len(response.detections)

    meshes = []
    for det, selected in zip(response.detections, bm):
        if selected:
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


def _render_polygons(response, label, *, post_process=False):
    if post_process:
        # get bboxes
        bboxes = []
        for det in response.detections:
            if det.roi.HasField("polygon"):
                x = [p.x for p in det.roi.polygon.points]
                y = [p.y for p in det.roi.polygon.points]
                bboxes.append([min(y), min(x), max(y), max(x)])

        bm = _filter_boxes(np.array(bboxes))

        detections = [
            det for det, selected in zip(response.detections, bm) if selected
        ]

    else:
        detections = response.detections

    c = len(detections)
    for det in reversed(detections):
        polygon = [[p.x, p.y] for p in det.roi.polygon.points]
        polygon = np.round(np.array(polygon)).astype(int)

        cv2.fillPoly(label, [polygon], c)
        c = c - 1

    return label


def _generate_label(response, label, *, post_process=False):
    if label.ndim == 2:
        _render_polygons(response, label, post_process=post_process)

    elif label.ndim == 3:
        _render_meshes(response, label, post_process=post_process)

    else:
        raise ValueError(
            f"supplied label template is not 2d or 3d: {label.shape}"
        )

    return label


def _adjust_response_offset(response, grid):
    for det in response.detections:
        for p in det.roi.polygon.points:
            p.x += grid[1].start
            p.y += grid[0].start
        for v in det.roi.mesh.verts:
            v.x += grid[2].start
            v.y += grid[1].start
            v.z += grid[0].start

    return response


@thread_worker
def grpc_object_detection(
    image_data: napari_data,
    settings: dict,
    grid_positions: list,
) -> Generator[np.ndarray, None, None]:
    is3d = settings["3D"]
    if is3d:
        assert image_data.ndim == 5
    else:
        assert image_data.ndim == 4

    # call server
    with _get_grpc_channel(settings) as channel:
        stub = proto.ObjectDetectionStub(channel)

        for image in image_data:
            # start with an empty response
            response = proto.DetectionResponse()

            for grid in grid_positions:
                logger.info(f"patch position {grid}")

                patch = np.array(image.__getitem__(grid))

                patch_response = stub.RunDetection(
                    _object_detection_build_request(patch, settings),
                    timeout=300 if settings["3D"] else 15,
                )

                patch_response = _adjust_response_offset(patch_response, grid)

                logger.info(
                    f"Detected {len(patch_response.detections)} cells in patch"
                )

                response.MergeFrom(patch_response)

                yield

            logger.info(f"Detected {len(response.detections)} cells in image")

            yield _generate_label(
                response, np.zeros(image_data.shape[1:-1], dtype="uint16")
            )


@thread_worker
def grpc_process_image(
    image_data: napari_data,
    settings: dict,
    grid_positions: list | None = None,
) -> Generator[np.ndarray, None, None]:
    is3d = settings["3D"]
    if is3d:
        assert image_data.ndim == 5
    else:
        assert image_data.ndim == 4

    assert (
        grid_positions is None
    ), "Grid-processing unimplemented -- try object-detection"

    # call server
    with _get_grpc_channel(settings) as channel:
        stub = proto.ProcessImageStub(channel)

        for image in image_data:
            response = stub.Run(
                proto.ProcessRequest(
                    image_data=proto.ImageData(pixels=_encode_image(image))
                ),
                timeout=300 if settings["3D"] else 15,
            )

            output = deserialize_to_numpy(response.image_data.pixels)
            if output.shape[-1] == 1:
                output = output.squeeze(-1)
            if not settings["3D"]:
                output = output.squeeze(0)

            yield output
