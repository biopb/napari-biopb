"""Shared tensor utilities for biopb-mcp.

Functions for building pyramid levels and determining dimension indices,
used by both the tensor browser widget and the MCP server.
"""

import logging
from typing import List, Optional, Tuple

from biopb.tensor import TensorFlightClient

from ._config import get_setting, load_config

logger = logging.getLogger(__name__)


def get_xy_dim_indices(tensor_desc) -> Tuple[int, int]:
    """Get indices of x and y dimensions from tensor descriptor.

    Uses dim_labels as primary source (looks for 'x', 'y').
    Falls back to the last two dimensions under the standard ``[..., Y, X]``
    convention (X last, Y second-to-last) when dim_labels are unavailable.

    Returns:
        Tuple of (y_index, x_index) - y first for row/col convention

    Raises:
        ValueError: the tensor has fewer than 2 dimensions (not a displayable
            image).
    """
    ndim = len(tensor_desc.shape)

    if tensor_desc.dim_labels:
        labels_lower = [l.lower() for l in tensor_desc.dim_labels]
        try:
            x_idx = labels_lower.index("x")
            y_idx = labels_lower.index("y")
            return (y_idx, x_idx)
        except ValueError:
            pass

    if ndim < 2:
        raise ValueError(
            f"Cannot identify x/y dimensions: tensor is {ndim}-D; napari needs "
            "at least 2 dimensions to display an image."
        )
    # Standard [..., Y, X]: X is the last axis, Y the second-to-last.
    return (ndim - 2, ndim - 1)


def build_pyramid_levels(
    client: TensorFlightClient,
    source_id: str,
    tensor_id: str,
    tensor_desc,
    config: Optional[dict] = None,
) -> List:
    """Build pyramid levels for large x-y datasets.

    ``threshold`` and ``downscale_factor`` come from the ``pyramid`` config
    section (``config`` defaults to the on-disk config). A pyramid is built only
    when an x/y dimension exceeds ``threshold``; levels are then emitted, each
    downsampled ``downscale_factor`` from the last, until the coarsest level
    fits within ``threshold`` in both x and y. Because the level before it
    exceeded ``threshold``, the coarsest level always lands in
    ``(threshold // downscale_factor, threshold]``.

    Returns:
        List of dask arrays at different resolution levels (pyramid)
    """
    if config is None:
        config = load_config()
    threshold = get_setting(config, "pyramid.threshold")
    downscale_factor = get_setting(config, "pyramid.downscale_factor")

    shape = tensor_desc.shape
    ndim = len(shape)

    y_idx, x_idx = get_xy_dim_indices(tensor_desc)

    x_size = shape[x_idx]
    y_size = shape[y_idx]

    if x_size <= threshold and y_size <= threshold:
        return [client.get_tensor(source_id, tensor_id)]

    levels = []
    scale = 1

    while True:
        scale_hint = [1] * ndim
        scale_hint[y_idx] = scale
        scale_hint[x_idx] = scale

        arr = client.get_tensor(source_id, tensor_id, scale_hint=scale_hint)
        levels.append(arr)

        # Stop once this level fits within the threshold in both x and y. The
        # previous level exceeded it, so the coarsest level lands in
        # (threshold // downscale_factor, threshold] -- no separate floor needed.
        if x_size // scale <= threshold and y_size // scale <= threshold:
            break

        scale *= downscale_factor

    return levels


def build_layer_scale(
    client: TensorFlightClient,
    source_id: str,
    tensor_desc,
    source_desc=None,
) -> Tuple[Optional[List[float]], Optional[dict]]:
    """Build a napari ``scale`` vector from a source's OME pixel sizes.

    Reads ``client.get_source_metadata`` (a dict — the server's OME model
    dumped to JSON) and maps ``physical_size_x/y/z`` onto the tensor's
    dimension axes, so areas/volumes the agent computes come out in physical
    units (e.g. µm²) instead of pixels.

    Axis order comes from ``tensor_desc.dim_labels``, falling back to the source
    descriptor's ``dim_labels`` (``source_desc``) when the per-tensor labels are
    empty, then to positional x/y.

    Returns:
        ``(scale, info)`` where *scale* is a per-axis list aligned to
        ``tensor_desc`` dims (``None`` if no physical sizes are available) and
        *info* is a small dict of the physical sizes + units for surfacing to
        the agent (``None`` if unavailable).
    """

    def _positive_float(value):
        """Coerce to a positive float, or None for missing/garbage values."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    try:
        metadata = client.get_source_metadata(source_id)

        images = metadata.get("images") if isinstance(metadata, dict) else None
        if not images:
            return None, None
        pixels = images[0].get("pixels")
        if not pixels:
            return None, None

        psize = {
            "x": _positive_float(pixels.get("physical_size_x")),
            "y": _positive_float(pixels.get("physical_size_y")),
            "z": _positive_float(pixels.get("physical_size_z")),
        }
        if not any(psize.values()):
            return None, None

        ndim = len(tensor_desc.shape)
        dim_labels = tensor_desc.dim_labels or getattr(
            source_desc, "dim_labels", None
        )
        labels = [str(label).lower() for label in (dim_labels or [])]

        scale = [1.0] * ndim
        for axis, value in psize.items():
            if value and axis in labels:
                scale[labels.index(axis)] = value

        # Fall back to the conventional trailing (..., y, x) axes when the
        # descriptor carries no usable labels.
        if "x" not in labels and "y" not in labels and ndim >= 2:
            if psize["x"]:
                scale[ndim - 1] = psize["x"]
            if psize["y"]:
                scale[ndim - 2] = psize["y"]

        info = {
            "physical_size_x": psize["x"],
            "physical_size_y": psize["y"],
            "physical_size_z": psize["z"],
            "physical_size_x_unit": pixels.get("physical_size_x_unit"),
            "physical_size_y_unit": pixels.get("physical_size_y_unit"),
            "physical_size_z_unit": pixels.get("physical_size_z_unit"),
        }
        return scale, info
    except Exception as exc:
        logger.warning("build_layer_scale failed for %s: %s", source_id, exc)
        return None, None


def add_tensor_layer(
    viewer,
    client: TensorFlightClient,
    source_id: str,
    tensor_id: str,
    tensor_desc,
    *,
    name: str,
    source_desc=None,
    compute_scheduler: Optional[str] = None,
    config: Optional[dict] = None,
):
    """Build a tensor's pyramid and add it to *viewer* as an image layer.

    The shared "load a tensor into the viewer" pipeline used by both the Tensor
    Browser widget and the MCP ``add_tensor``: build pyramid levels, pin their
    slice reads to a single-process scheduler so the serial viewer shares the
    main-process chunk cache (issue #8; no-op standalone), attach the source's
    OME physical pixel size as ``scale`` + ``metadata['ome_physical_size']`` so
    the agent's areas/volumes come out in physical units, then ``add_image``
    (``multiscale=True`` when there is more than one level).

    Source resolution, layer *name*, and any cursor/logging/error handling stay
    with the caller; everything from building levels through ``add_image`` is
    uniform here so the three call sites can't drift.

    Returns the created napari layer.
    """
    from ._viewer_compute import wrap_levels

    levels = build_pyramid_levels(
        client, source_id, tensor_id, tensor_desc, config=config
    )
    levels = wrap_levels(levels, compute_scheduler)

    add_kwargs = {"name": name}
    scale, phys = build_layer_scale(
        client, source_id, tensor_desc, source_desc=source_desc
    )
    if scale is not None:
        add_kwargs["scale"] = scale
    if phys is not None:
        add_kwargs["metadata"] = {"ome_physical_size": phys}

    if len(levels) > 1:
        return viewer.add_image(levels, multiscale=True, **add_kwargs)
    return viewer.add_image(levels[0], **add_kwargs)
