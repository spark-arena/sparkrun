"""Script generators for remote execution.

Each function generates a complete bash script as a string.
These scripts are fed to remote hosts via ``ssh host bash -s``.
"""

from __future__ import annotations

from sparkrun.scripts import read_script


def generate_ip_detect_script() -> str:
    """Generate a script that detects the host's management IP address.

    The detected IP is printed as the last line of stdout.

    Returns:
        Bash script content as a string.
    """
    return read_script("ip_detect.sh")
