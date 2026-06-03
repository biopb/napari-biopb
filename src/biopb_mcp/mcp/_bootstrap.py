"""Bootstrap executed *inside* the MCP child kernel.

Injected via IPython ``exec_lines`` so it runs before the kernel services any
tool calls.  It enables the Qt event loop, configures dask in the process
where compute actually happens, opens a visible napari viewer with the Tensor
Browser widget, and populates the ``execute_code`` namespace.

A failure here does not abort the kernel (exec_lines errors are swallowed by
IPython), so ``bootstrap`` prints a ``BOOTSTRAP_ERROR`` sentinel that the
host's health probe detects via the absence of ``viewer`` in the namespace.
"""

import logging
import traceback

logger = logging.getLogger(__name__)


def _configure_dask(mcp_config: dict):
    """Set up dask in the kernel process.

    Returns ``(client, cluster)``:

    * ``"distributed"`` + an external ``dask_distributed_address`` -> a
      ``Client`` attached to that scheduler; ``cluster`` is ``None``.
    * ``"distributed"`` + no address -> a kernel-local multi-process
      ``LocalCluster`` and a ``Client`` bound to it. This is the default and
      the only mode where ``cancel_job`` can stop an in-flight ``compute()``.
    * ``"threads"`` / ``"synchronous"`` -> in-process scheduler; both ``None``.

    A failure spinning the local cluster degrades gracefully to the in-process
    ``threads`` scheduler rather than aborting the bootstrap.
    """
    import dask

    scheduler = mcp_config.get("dask_scheduler", "distributed")
    num_workers = mcp_config.get("dask_num_workers", 0) or None
    address = mcp_config.get("dask_distributed_address", "")

    if scheduler == "distributed":
        try:
            from dask.distributed import Client

            if address:
                client = Client(address)
                logger.info("Dask using distributed scheduler at %s", address)
                return client, None

            from dask.distributed import LocalCluster

            cluster = LocalCluster(
                n_workers=num_workers,
                processes=True,
                threads_per_worker=mcp_config.get(
                    "dask_threads_per_worker", 1
                ),
                memory_limit=mcp_config.get("dask_memory_limit", "auto"),
                dashboard_address=mcp_config.get(
                    "dask_dashboard_address", "127.0.0.1:0"
                ),
            )
            client = Client(cluster)
            logger.info(
                "Dask using local cluster: %d worker(s) at %s",
                len(cluster.workers),
                cluster.scheduler_address,
            )
            return client, cluster
        except Exception:
            # Covers a missing `distributed` install, an unreachable external
            # address, or a LocalCluster spawn failure -- degrade to the
            # in-process scheduler so the bootstrap (and the viewer) survives.
            logger.exception(
                "Distributed dask unavailable; "
                "falling back to in-process threads scheduler"
            )
            scheduler = "threads"

    dask.config.set(scheduler=scheduler, num_workers=num_workers)
    logger.info("Dask scheduler: %s, num_workers: %s", scheduler, num_workers)
    return None, None


def bootstrap():
    """Entry point called from the kernel's exec_lines."""
    try:
        _bootstrap_impl()
    except Exception:
        print("BOOTSTRAP_ERROR: " + traceback.format_exc())


def _bootstrap_impl():
    import dask.array as da
    import napari
    import numpy as np
    from IPython import get_ipython

    from .._config import load_config
    from .._connection import TensorConnection
    from ..tensor_browser import TensorBrowserWidget
    from . import _jobs
    from ._helpers import patch_viewer_load_tensor
    from ._process_ops import build_ops

    # 1. Qt integration must be enabled before the viewer is created so napari
    #    shares the kernel's integrated Qt event loop (programmatic %gui qt).
    ip = get_ipython()
    ip.enable_gui("qt")

    config = load_config()
    mcp_config = config.get("mcp", {})

    # 2. Configure dask in the compute process.
    dask_client, dask_cluster = _configure_dask(mcp_config)

    # 3. Data-access service, shared by the widget and the agent namespace.
    conn = TensorConnection(config)

    # 4. Visible napari viewer + Tensor Browser (auto-connects on its own tick).
    viewer = napari.Viewer()
    tbw = TensorBrowserWidget(viewer, connection=conn)
    viewer.window.add_dock_widget(tbw, name="Tensor Browser")

    # 5. ProcessImage ops: thin Run() callables for each configured servicer.
    #    client_getter reads conn.client lazily so the async-connecting tensor
    #    client is picked up at call time.
    timeout_config = config.get("timeout", {})
    grpc_config = config.get("grpc", {})
    max_msg_bytes = grpc_config.get("max_message_size_mb", 512) * 1024 * 1024
    channel_options = [
        ("grpc.max_receive_message_length", max_msg_bytes),
        ("grpc.max_send_message_length", max_msg_bytes),
    ]
    try:
        ops = build_ops(
            client_getter=lambda: conn.client,
            server_urls=mcp_config.get("process_image_servers", []),
            op_names_timeout=timeout_config.get("get_op_names", 10.0),
            run_timeout=timeout_config.get("process_image", 300.0),
            channel_options=channel_options,
        )
    except Exception:
        logger.exception("Failed to build ProcessImage ops")
        ops = {}

    # 6. Async job runner: execute_code runs in a background kernel thread so
    #    the main thread / Qt loop stays free for screenshot/status mid-job.
    #    install() stores the shell, installs the thread-aware stdout streams,
    #    and clears any prior job state; wrap_viewer_for_threads marshals the
    #    common viewer-mutating methods to the Qt main thread.
    patch_viewer_load_tensor(viewer, conn)
    _jobs.install(ip)
    _jobs.wrap_viewer_for_threads(viewer)

    # 7. Namespace for execute_code.  client is refreshed per-job by the job
    #    runner (the connection service connects asynchronously).
    ip.user_ns.update(
        {
            "viewer": viewer,
            "np": np,
            "da": da,
            "client": None,
            "ops": ops,
            "_conn": conn,
            "_dask_client": dask_client,
            "_dask_cluster": dask_cluster,
            "_jobs": _jobs,
            "run_on_main": _jobs.run_on_main,
            "cancelled": _jobs.cancelled,
        }
    )
