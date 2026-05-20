"""Helpers for loading embedded package resources via :mod:`importlib.resources`.

This module provides a single generic loader (:func:`load_resource`) and two
convenience wrappers used throughout sparkrun:

* :func:`load_yaml_resource` — load and parse a YAML file bundled inside a
  package directory.
* :func:`load_script_resource` — load a bash (or Python) script from the
  :mod:`sparkrun.scripts` package.

All functions raise :class:`FileNotFoundError` (propagated from
``importlib.resources``) when the requested resource does not exist.
"""

from __future__ import annotations

from importlib.resources import files


def load_resource(package: str, name: str, *, encoding: str = "utf-8") -> str:
    """Load an embedded package resource as a string.

    Args:
        package: Fully-qualified package name whose directory contains the
            resource (e.g. ``"sparkrun.scripts"``).
        name: Filename of the resource within that package directory
            (e.g. ``"ip_detect.sh"``).
        encoding: Text encoding used to decode the resource bytes.
            Defaults to ``"utf-8"``.

    Returns:
        The resource content as a :class:`str`.

    Raises:
        FileNotFoundError: If the resource does not exist inside *package*.
    """
    return files(package).joinpath(name).read_text(encoding=encoding)


def load_yaml_resource(package: str, name: str) -> dict:
    """Load and parse a YAML resource bundled inside *package*.

    Args:
        package: Fully-qualified package name containing the YAML file.
        name: Filename of the YAML resource (e.g. ``"defaults.yaml"``).

    Returns:
        Parsed YAML content as a :class:`dict`, or an empty dict when the
        top-level YAML value is not a mapping.

    Raises:
        FileNotFoundError: If the resource does not exist inside *package*.
    """
    import yaml

    data = yaml.safe_load(load_resource(package, name))
    return data if isinstance(data, dict) else {}


def load_script_resource(name: str) -> str:
    """Load a bash or Python script from the :mod:`sparkrun.scripts` package.

    This is a convenience wrapper around :func:`load_resource` that hard-codes
    the *package* argument to ``"sparkrun.scripts"``.

    Args:
        name: Script filename (e.g. ``"ip_detect.sh"``).

    Returns:
        Script content as a :class:`str`.

    Raises:
        FileNotFoundError: If the script does not exist inside
            ``sparkrun.scripts``.
    """
    return load_resource("sparkrun.scripts", name)
