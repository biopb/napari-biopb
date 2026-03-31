import logging
from typing import Generator

import biopb.image as proto
import grpc
import numpy as np
from biopb.image.utils import deserialize_to_numpy, serialize_from_numpy
from google.protobuf import struct_pb2
from grpc_health.v1 import health_pb2, health_pb2_grpc
from napari.qt.threading import thread_worker

from ._render import _adjust_response_offset, _generate_label
from ._typing import napari_data

logger = logging.getLogger(__name__)


def dict_to_struct(d: dict) -> struct_pb2.Struct:
    """Convert Python dict to protobuf Struct."""
    s = struct_pb2.Struct()
    s.update(d)
    return s


def struct_to_dict(s: struct_pb2.Struct) -> dict:
    """Convert protobuf Struct to Python dict."""
    return dict(s)


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


def check_server_health(settings: dict, timeout: float = 5.0) -> bool:
    """Check if the gRPC server is healthy and ready to serve requests.

    Args:
        settings: Widget settings dict (must contain 'Server' and 'Scheme')
        timeout: Timeout in seconds for the health check

    Returns:
        True if server is healthy, False otherwise
    """
    try:
        with _get_grpc_channel(settings) as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest()
            response = stub.Check(request, timeout=timeout)
            return response.status == health_pb2.HealthCheckResponse.SERVING
    except Exception as e:
        logger.debug("Health check failed: %s", e)
        return False


def get_op_names(settings: dict, timeout: float = 10.0) -> proto.OpNames:
    """Query server for available operations and their schemas.

    Args:
        settings: Widget settings dict (must contain 'Server' and 'Scheme')
        timeout: Timeout in seconds for the request

    Returns:
        OpNames proto containing operation names and their schemas

    Raises:
        Exception: If connection or query fails
    """
    with _get_grpc_channel(settings) as channel:
        stub = proto.ProcessImageStub(channel)
        response = stub.GetOpNames(proto.GetOpNamesRequest(), timeout=timeout)
        return response


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

    server = settings["Server"]
    logger.info("Starting object detection on %s", server)

    # call server
    with _get_grpc_channel(settings) as channel:
        stub = proto.ObjectDetectionStub(channel)

        for image in image_data:
            # start with an empty response
            response = proto.DetectionResponse()

            for grid in grid_positions:
                logger.debug("Processing patch %s", grid)

                patch = np.array(image.__getitem__(grid))

                patch_response = stub.RunDetection(
                    _object_detection_build_request(patch, settings),
                    timeout=300 if settings["3D"] else 15,
                )

                patch_response = _adjust_response_offset(patch_response, grid)

                logger.debug(
                    "Detected %d cells in patch",
                    len(patch_response.detections),
                )

                response.MergeFrom(patch_response)

                yield

            logger.info("Detected %d cells in image", len(response.detections))

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
        settings: Widget settings dict (includes 'Op' and kwargs)
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

    server = settings["Server"]
    logger.info("Starting image processing on %s", server)

    # Extract op_name and kwargs from settings
    op_name = settings.get("Op", "")
    kwargs = _extract_kwargs(settings)

    # call server
    with _get_grpc_channel(settings) as channel:
        stub = proto.ProcessImageStub(channel)

        for image in image_data:
            image = np.array(image)
            response = stub.Run(
                proto.ProcessRequest(
                    image_data=proto.ImageData(pixels=_encode_image(image)),
                    op_name=op_name,
                    kwargs=dict_to_struct(kwargs),
                ),
                timeout=300 if settings["3D"] else 15,
            )

            output = deserialize_to_numpy(response.image_data.pixels)
            # if output.shape[-1] == 1:
            #     output = output.squeeze(-1)
            if not settings["3D"]:
                output = output.squeeze(0)

            logger.debug("Processed image, output shape: %s", output.shape)
            yield output


def _extract_kwargs(settings: dict) -> dict:
    """Extract kwargs from widget settings.

    Settings keys matching known op kwarg names are extracted.
    Non-op-related keys (Image, 3D, Server, Scheme, Status, Op) are excluded.
    """
    exclude_keys = {"Image", "3D", "Server", "Scheme", "Status", "Op"}
    kwargs = {}
    for key, value in settings.items():
        if key not in exclude_keys:
            kwargs[key] = value
    return kwargs
