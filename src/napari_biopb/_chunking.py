"""Chunking utilities for dimensional iteration based on InputShapeHint."""

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

if TYPE_CHECKING:
    import napari
    import biopb.image as proto

logger = logging.getLogger(__name__)


class IterationSpec(NamedTuple):
    iter_dims: list  # Numeric dims to iterate (empty if no iteration)
    axis_order: str  # Full axis order for serialization (e.g., "TZYXC")


def _get_axis_mapping(image_layer: "napari.layers.Image", is_3d: bool) -> str:
    """Return axis order string respecting napari's interpretation.

    Uses layer.rgb, shape, and dtype to determine axis semantics.
    Handles all valid napari shapes (2-5 dims after multiscale handling).

    Args:
        image_layer: napari Image layer (uses .rgb, .multiscale, .data properties)
        is_3d: user's 3D checkbox setting (determines Z interpretation)

    Returns:
        Axis order string (e.g., "YX", "YXC", "ZYX", "TZYXC")

    Raises:
        ValueError: If ndim not in [2, 3, 4, 5] after multiscale handling
    """
    # Handle multiscale: use highest resolution level
    data = image_layer.data
    if image_layer.multiscale:
        data = data[0]
        logger.debug(
            f"Image is multiscale, using highest resolution level for axis mapping. Shape: {data.shape}"
        )

    ndim = data.ndim
    if ndim not in (2, 3, 4, 5):
        raise ValueError(f"Data must have 2-5 dimensions, got {ndim}")

    is_rgb = image_layer.rgb
    if ndim == 2:
        assert not is_rgb, "RGB/RGBA cannot be 2D arrays"
        channels = "YX"
    elif ndim == 3:
        channels = "YXC" if is_rgb else "ZYX" if is_3d else "TYX"
    elif ndim == 4:
        if is_rgb:
            channels = "ZYXC" if is_3d else "TYXC"
        else:
            channels = "TZYX" if is_3d else "TCYX"
    else:
        channels = "TZYXC" if is_rgb else "TZCYX"

    logger.debug(
        f"Determined axis order '{channels}' for data shape {data.shape} with rgb={is_rgb} and is_3d={is_3d}"
    )

    return channels


def _validate_data_shape(
    data: np.ndarray,
    axis_order: str,
    hint: Optional["proto.InputShapeHint"],
) -> None:
    """Validate data shape against hint, return warnings.

    Args:
        data: Image data array
        axis_order: Axis order string (e.g., "YXC")
        hint: InputShapeHint from op schema (may be None)
    """
    if hint is None:
        return

    for axis_name in hint.required_multivalue:
        axis_name = str(axis_name).upper()  # Ensure uppercase for comparison
        if axis_name not in axis_order:
            logger.warning(
                f"Service requires multivalue axis '{axis_name}' but it's not present in data"
            )
        elif data.shape[axis_order.index(axis_name)] == 1:
            logger.warning(
                f"Dimension '{axis_name}' has size 1 but service requires >1"
            )


def _get_iter_spec(
    axis_order: str,
    hint: Optional["proto.InputShapeHint"],
) -> IterationSpec:
    """Compute iteration dims and submitting order.

    Args:
        axis_order: Axis order string (e.g., "YXC")
        hint: InputShapeHint from op schema (may be None)

    Returns:
        IterationSpec with iter_dims and axis_order
    """
    submission_axes = set("ZYXC")

    # Step 1: Remove axes from expected_singletons (but never remove X or Y)
    if hint is not None and hint.expected_singletons:
        submission_axes -= set(hint.expected_singletons) - {"X", "Y"}

    # Step 2: Add required_multivalue axes to submission
    if hint is not None and hint.required_multivalue:
        submission_axes |= set(hint.required_multivalue)

    # Step 3: Remove any axes not present in axis_order
    submission_axes &= set(axis_order)

    logger.debug(f"_get_iter_spec: submission_axes='{submission_axes}'")

    # Step 4: Remaining axes are iter_dims
    iter_axes = set(axis_order) - set(submission_axes)
    iter_dims = sorted([axis_order.index(ax) for ax in iter_axes])
    logger.debug(f"_get_iter_spec: iter_axes={iter_axes}")

    return IterationSpec(
        iter_dims=iter_dims,
        axis_order=axis_order,
    )


def _data_iterator(data: np.ndarray, iter_dims: Optional[list]):
    """Iterate over data slices for given dims.

    Args:
        data: Image data array
        iter_dims: List of dim indices to iterate, or None/empty

    Yields:
        (position_dict, sliced_data) where position_dict maps iter_dims to indices
    """
    if not iter_dims:
        yield {}, data
        return

    # Build shape for iteration dims only
    iter_shape = tuple(data.shape[d] for d in iter_dims)

    for idx_tuple in np.ndindex(iter_shape):
        position = dict(zip(iter_dims, idx_tuple))

        # Build slices: iterate dims get single index, others get full slice
        slices = [slice(None)] * data.ndim
        for dim_idx, idx in position.items():
            slices[dim_idx] = slice(idx, idx + 1)

        yield position, data[tuple(slices)]


class ResultBuilder:
    """Lazy result buffer construction."""

    def __init__(self, original_shape: tuple, iter_dims: Optional[list]):
        self.original_shape = original_shape
        self.iter_dims = iter_dims
        self.buffer = None
        self.dtype = None

    def add_result(
        self, position: dict, chunk: np.ndarray, squeezed_dims: list
    ):
        """Add result chunk to buffer.

        Args:
            position: Dict mapping iter_dims to indices
            chunk: Result chunk array (may have squeezed dims removed)
            squeezed_dims: Dims that were squeezed before sending
        """
        logger.debug(
            f"Adding chunk with shape {chunk.shape} at position {position} and squeezed_dims {squeezed_dims}"
        )

        if self.buffer is None:
            output_shape = self._infer_output_shape(chunk, squeezed_dims)
            self.dtype = chunk.dtype
            self.buffer = np.zeros(output_shape, dtype=self.dtype)

        placement_slices = self._build_placement_slices(
            position, squeezed_dims
        )
        self.buffer[placement_slices] = chunk

    def _infer_output_shape(
        self, chunk: np.ndarray, squeezed_dims: list
    ) -> tuple:
        """Infer output buffer shape from first chunk.

        The output shape uses:
        - original_shape for iterated dimensions (we place each result at position)
        - chunk.shape for non-iterated dimensions (server output may differ)
        """
        output_shape = []
        for i, (orig_dim, chunk_dim) in enumerate(
            zip(self.original_shape, chunk.shape)
        ):
            if i in squeezed_dims:
                output_shape.append(
                    orig_dim
                )  # Iter dim: preserve original size
            else:
                output_shape.append(
                    chunk_dim
                )  # Non-iter: use server output size
        return tuple(output_shape)

    def _build_placement_slices(
        self, position: dict, squeezed_dims: list
    ) -> tuple:
        """Build slice tuple for placing chunk in buffer."""
        slices = []
        for i in range(len(self.original_shape)):
            if i in position:
                # Iteration dim: place at specific index
                slices.append(slice(position[i], position[i] + 1))
            else:
                # Non-iteration dim: full slice
                slices.append(slice(None))

        return tuple(slices)

    def get_result(self) -> np.ndarray:
        """Return final assembled result."""
        return self.buffer
