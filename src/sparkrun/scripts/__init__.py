"""Embedded bash scripts for remote execution.

Scripts are stored as .sh files alongside this module and loaded
via :func:`read_script` at runtime.
"""

from __future__ import annotations

from importlib import resources

from sparkrun.utils.resource_loader import load_resource


def read_script(name: str) -> str:
    """Read a bash script from the scripts package.

    Args:
        name: Script filename (e.g. ``"ip_detect.sh"``).

    Returns:
        Script content as a string.
    """
    return load_resource(__package__, name)


def get_script_path(name: str):
    """Return a context manager that yields a filesystem :class:`~pathlib.Path` for a script.

    Usage::

        with get_script_path("mesh_ssh_keys.sh") as path:
            subprocess.run(["bash", str(path), ...])

    The context manager guarantees the path exists on disk even when the
    package is installed inside a zip archive.
    """
    return resources.as_file(resources.files(__package__).joinpath(name))
