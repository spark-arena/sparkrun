"""Tests for the workload-log signature detector.

The detector's whole value is its false-positive rate: it recommends running a
container as root, which is advice that must not appear for problems root does
not fix.  Most of these tests are therefore negatives.
"""

from __future__ import annotations

import pytest

from sparkrun.utils.log_diagnostics import (
    EACCES,
    ENOENT,
    EROFS,
    detect_in_place_write_failure,
)

#: The two lines the detector actually reads, in the shape real output has
#: them: a directory-creation frame (the ENOENT gate) and an errno line whose
#: path is under a Python installation tree. Synthesized rather than a captured
#: crash log — the surrounding frames are decoration, and a verbatim dump would
#: be a fixture nobody can safely trim later.
#:
#: The ``(EngineCore_DP0 pid=NNN)`` prefix is deliberate: vLLM prefixes every
#: worker line, so a future anchored regex would silently stop matching real
#: logs while a bare traceback fixture kept passing.
ENOENT_MKDIR_LOG = (
    '(EngineCore_DP0 pid=481)   File "/usr/lib/python3.12/pathlib.py", line 1313, in mkdir\n'
    "(EngineCore_DP0 pid=481)     os.mkdir(self, mode)\n"
    "(EngineCore_DP0 pid=481) FileNotFoundError: [Errno 2] No such file or directory: "
    "'/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/cutlass_instantiations/120_mxfp4min'\n"
)


class TestDetectsRealSignatures:
    def test_issue_280_traceback(self):
        """The failure this whole path exists to attribute (issue #280).

        The eugr ``vllm-node-mxfp4`` image pins a flashinfer fork that generates
        cutlass instantiations into its own ``dist-packages`` — fine as root,
        impossible under sparkrun's rootless ``--user $(id -u):$(id -g)``.
        """
        failure = detect_in_place_write_failure(ENOENT_MKDIR_LOG)

        assert failure is not None
        assert failure.errno == ENOENT
        assert failure.message == "No such file or directory"
        assert failure.path.endswith("cutlass_instantiations/120_mxfp4min")

    def test_permission_denied(self):
        failure = detect_in_place_write_failure(
            "PermissionError: [Errno 13] Permission denied: '/opt/venv/lib/python3.12/site-packages/foo/gen'\n"
        )
        assert failure is not None
        assert failure.errno == EACCES
        assert failure.message == "Permission denied"

    def test_read_only_file_system(self):
        failure = detect_in_place_write_failure("OSError: [Errno 30] Read-only file system: '/usr/lib/python3/dist-packages/x/y'\n")
        assert failure is not None
        assert failure.errno == EROFS
        assert failure.message == "Read-only file system"

    @pytest.mark.parametrize("frame", ["    os.makedirs(path)", '  File "x.py", line 1, in makedirs', '  File "x.py", line 1, in _mkdir'])
    def test_any_creation_frame_admits_enoent(self, frame):
        text = "%s\nFileNotFoundError: [Errno 2] No such file or directory: '/x/site-packages/pkg/gen/a'\n" % frame
        assert detect_in_place_write_failure(text) is not None

    def test_reports_the_first_match(self):
        """Crash logs repeat the same traceback once per worker rank.

        Reporting the last one would name a different rank's copy of the same
        failure, which reads as though two things went wrong.
        """
        one = "PermissionError: [Errno 13] Permission denied: '/x/site-packages/a'\n"
        two = "PermissionError: [Errno 13] Permission denied: '/x/site-packages/b'\n"
        failure = detect_in_place_write_failure(one + two)
        assert failure is not None and failure.path.endswith("/a")


class TestDoesNotFire:
    """Everything root would not fix, or would not fix *for that reason*."""

    def test_missing_package_data_file_is_an_image_defect(self):
        """ENOENT under site-packages with no creation frame.

        A packaged data file that isn't there is a broken image; running as
        root does not conjure it, and saying so would be actively misleading.
        """
        text = (
            '  File "/opt/venv/lib/python3.12/site-packages/pkg/loader.py", line 9, in load\n'
            "    return open(p).read()\n"
            "FileNotFoundError: [Errno 2] No such file or directory: '/opt/venv/lib/python3.12/site-packages/pkg/data/weights.json'\n"
        )
        assert detect_in_place_write_failure(text) is None

    def test_write_failure_outside_the_install_tree(self):
        """A read-only bind mount or an unwritable cache dir is a different bug.

        ``-o user=root`` might even paper over it while leaving root-owned
        files behind, so the path gate applies to every errno uniformly.
        """
        text = "PermissionError: [Errno 13] Permission denied: '/cache/runtime/flashinfer/x'\n"
        assert detect_in_place_write_failure(text) is None

    def test_unrelated_errno_under_the_install_tree(self):
        assert detect_in_place_write_failure("OSError: [Errno 28] No space left on device: '/x/site-packages/a'\n") is None

    def test_ordinary_engine_crash(self):
        text = (
            "Traceback (most recent call last):\n"
            '  File "/opt/venv/lib/python3.12/site-packages/vllm/engine.py", line 42, in _init\n'
            "    raise ValueError('No available memory for the cache blocks')\n"
            "ValueError: No available memory for the cache blocks\n"
        )
        assert detect_in_place_write_failure(text) is None

    @pytest.mark.parametrize("text", ["", None, "   \n", "INFO: server started\n"])
    def test_empty_and_benign_input(self, text):
        assert detect_in_place_write_failure(text) is None
