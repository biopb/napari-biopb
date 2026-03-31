import logging
from typing import Generator

import biopb.image as proto
import grpc
import numpy as np
from biopb.image.utils import deserialize_to_numpy, serialize_from_numpy
from napari.qt.threading import thread_worker

from ._render import _adjust_response_offset, _generate_label
from ._typing import napari_data

logger = logging.getLogger(__name__)


def _encode_image(image: np.ndarray, z_ratio: float = 1.0):
    """Encode numpy image array to protobuf Pixels format."""
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
    """Serialize a np image array as ImageData protobuf."""
    pixels = _encode_image(image, settings["Z Aspect Ratio"])

    request = proto.DetectionRequest(
        image_data=proto.ImageData(pixels=pixels),
        detection_settings=_get_detection_settings(settings),
    )

    return request


def _get_grpc_channel(settings: dict):
    """Create gRPC channel based on server settings."""
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
    """Convert widget settings to DetectionSettings protobuf."""
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


@thread_worker
def grpc_object_detection(
    image_data: napari_data,
    settings: dict,
    grid_positions: list,
) -> Generator[np.ndarray, None, None]:
    """Run object detection on image data via gRPC.

    Args:
        image_data: Input image(s) as dask array or numpy array
        settings: Widget settings dict
        grid_positions: List of slice tuples for patch positions

    Yields:
        None for progress updates, then label array for each image
    """
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
    """Run image processing via gRPC.

    Args:
        image_data: Input image(s) as dask array or numpy array
        settings: Widget settings dict
        grid_positions: Not implemented for image processing

    Yields:
        Processed image array for each input image
    """
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
            image = np.array(image)
            response = stub.Run(
                proto.ProcessRequest(
                    image_data=proto.ImageData(pixels=_encode_image(image))
                ),
                timeout=300 if settings["3D"] else 15,
            )

            output = deserialize_to_numpy(response.image_data.pixels)
            # if output.shape[-1] == 1:
            #     output = output.squeeze(-1)
            if not settings["3D"]:
                output = output.squeeze(0)

            yield output
