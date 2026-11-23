"""Script generators for remote execution.

Each function generates a complete bash script as a string.
These scripts are fed to remote hosts via ``ssh host bash -s``.
"""

from __future__ import annotations

from sparkrun.scripts import inject_mgmt_vars, read_script


def generate_ip_detect_script(mgmt_interface: str | None = None, peers: list[str] | None = None) -> str:
    """Generate a script that detects the host's management IP address.

    The detected IP is printed as the last line of stdout.

    Args:
        mgmt_interface: The cluster's pinned management interface, if any.
        peers: The cluster's hosts, against which the host finds its
            management interface (see :func:`sparkrun.scripts.inject_mgmt_vars`).

    Returns:
        Bash script content as a string.
    """
    return inject_mgmt_vars(read_script("ip_detect.sh"), mgmt_interface=mgmt_interface, peers=peers)
