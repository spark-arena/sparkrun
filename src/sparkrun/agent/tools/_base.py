"""Base tool class for sparkrun agent tools."""

from __future__ import annotations

import logging
import shutil
import subprocess

from smolagents import Tool

logger = logging.getLogger(__name__)


class SparkrunBaseTool(Tool):
    """Base class for tools that invoke the sparkrun CLI.

    Provides a helper to shell out to ``sparkrun`` and capture output.
    """

    def _run_sparkrun(self, *args: str, timeout: int = 120) -> str:
        """Run a sparkrun CLI command and return combined stdout/stderr.

        Args:
            *args: Arguments to pass to ``sparkrun``.
            timeout: Command timeout in seconds.

        Returns:
            Combined stdout and stderr as a string.

        Raises:
            RuntimeError: If sparkrun is not found on PATH.
        """
        sparkrun_bin = shutil.which("sparkrun")
        if not sparkrun_bin:
            raise RuntimeError("sparkrun not found on PATH")

        cmd = [sparkrun_bin, *args]
        logger.debug("Running: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout
        if result.stderr:
            output = output + "\n" + result.stderr if output else result.stderr

        if result.returncode != 0:
            return "Command failed (exit %d):\n%s" % (result.returncode, output)

        return output.strip() if output else "(no output)"
