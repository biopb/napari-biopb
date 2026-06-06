"""Tests for the launcher's transport selection and kernel-log helper.

These exercise the pure plumbing in ``biopb_mcp.mcp.__main__`` (arg parsing and
the stdio kernel-log file) without starting a real kernel or viewer.
"""

import sys

import pytest

from biopb_mcp.mcp.__main__ import _open_kernel_log, _parse_args


class TestParseArgs:
    def test_defaults_come_from_config(self):
        opts = _parse_args([], default_transport="http", default_port=8765)
        assert opts.transport == "http"
        assert opts.port == 8765

    def test_config_default_can_be_stdio(self):
        opts = _parse_args([], default_transport="stdio", default_port=8765)
        assert opts.transport == "stdio"

    def test_transport_flag_overrides(self):
        opts = _parse_args(
            ["--transport", "stdio"], default_transport="http", default_port=1
        )
        assert opts.transport == "stdio"

    def test_port_flag_overrides(self):
        opts = _parse_args(
            ["--port", "9000"], default_transport="http", default_port=8765
        )
        assert opts.port == 9000

    def test_unknown_transport_rejected(self):
        with pytest.raises(SystemExit):
            _parse_args(
                ["--transport", "ftp"],
                default_transport="http",
                default_port=8765,
            )


class TestOpenKernelLog:
    def test_uses_configured_path(self, tmp_path):
        path = tmp_path / "k.log"
        f = _open_kernel_log({"kernel_log": str(path)})
        try:
            f.write("hello\n")
            f.flush()
        finally:
            f.close()
        assert path.read_text() == "hello\n"

    def test_empty_path_defaults_under_config_dir(self, tmp_path, monkeypatch):
        # _open_kernel_log does `from .._config import get_config_dir` at call
        # time, so patching the source module is what takes effect.
        import biopb_mcp._config as cfg

        monkeypatch.setattr(cfg, "get_config_dir", lambda: tmp_path)

        f = _open_kernel_log({"kernel_log": ""})
        try:
            assert (tmp_path / "kernel.log").exists()
        finally:
            f.close()

    def test_falls_back_to_stderr_on_open_error(self):
        # An unwritable path must not crash the launcher; it degrades to stderr.
        f = _open_kernel_log(
            {"kernel_log": "/nonexistent_dir/deep/path/kernel.log"}
        )
        assert f is sys.stderr
