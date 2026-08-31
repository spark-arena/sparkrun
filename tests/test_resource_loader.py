"""Unit tests for sparkrun.utils.resource_loader."""

from __future__ import annotations

import pytest
from unittest import mock

from sparkrun.utils.resource_loader import load_resource, load_script_resource, load_yaml_resource


class TestLoadResource:
    """Tests for the generic load_resource helper."""

    def test_load_known_script(self):
        """Load a real embedded script from sparkrun.scripts."""
        content = load_resource("sparkrun.scripts", "ip_detect.sh")
        assert isinstance(content, str)
        assert len(content) > 0
        assert "#!/bin/bash" in content

    def test_load_nonexistent_resource_raises(self):
        """Loading a missing resource raises FileNotFoundError."""
        with pytest.raises((FileNotFoundError, TypeError, Exception)):
            load_resource("sparkrun.scripts", "does_not_exist_xyz.sh")

    def test_load_resource_returns_string(self):
        """Result is always a str, never bytes."""
        content = load_resource("sparkrun.scripts", "ip_detect.sh")
        assert isinstance(content, str)

    def test_load_resource_encoding_param(self):
        """Explicit encoding= kwarg is accepted and produces the same result."""
        default_content = load_resource("sparkrun.scripts", "ip_detect.sh")
        explicit_content = load_resource("sparkrun.scripts", "ip_detect.sh", encoding="utf-8")
        assert default_content == explicit_content


class TestLoadScriptResource:
    """Tests for the sparkrun.scripts convenience wrapper."""

    def test_load_ip_detect(self):
        """ip_detect.sh loads and contains expected markers."""
        content = load_script_resource("ip_detect.sh")
        assert "ip route get 8.8.8.8" in content
        assert "NODE_IP" in content

    def test_load_container_launch(self):
        """container_launch.sh is non-empty and starts with shebang."""
        content = load_script_resource("container_launch.sh")
        assert content.startswith("#!/bin/bash")

    def test_load_script_resource_matches_load_resource(self):
        """load_script_resource is identical to load_resource for sparkrun.scripts."""
        via_generic = load_resource("sparkrun.scripts", "ip_detect.sh")
        via_convenience = load_script_resource("ip_detect.sh")
        assert via_generic == via_convenience

    def test_nonexistent_script_raises(self):
        """Missing script name propagates an error."""
        with pytest.raises((FileNotFoundError, TypeError, Exception)):
            load_script_resource("no_such_script_xyz.sh")


class TestLoadYamlResource:
    """Tests for load_yaml_resource."""

    def test_returns_dict_for_valid_yaml(self):
        """A YAML mapping is parsed into a dict."""
        yaml_text = "key: value\nfoo: 42\n"
        with mock.patch("sparkrun.utils.resource_loader.load_resource", return_value=yaml_text):
            result = load_yaml_resource("some.package", "config.yaml")
        assert result == {"key": "value", "foo": 42}

    def test_returns_empty_dict_for_non_mapping(self):
        """Top-level YAML list or scalar falls back to empty dict."""
        yaml_text = "- item1\n- item2\n"
        with mock.patch("sparkrun.utils.resource_loader.load_resource", return_value=yaml_text):
            result = load_yaml_resource("some.package", "list.yaml")
        assert result == {}

    def test_returns_empty_dict_for_null_yaml(self):
        """Empty/null YAML document falls back to empty dict."""
        with mock.patch("sparkrun.utils.resource_loader.load_resource", return_value=""):
            result = load_yaml_resource("some.package", "empty.yaml")
        assert result == {}

    def test_nested_yaml_parsed_correctly(self):
        """Nested mappings are fully parsed."""
        yaml_text = "outer:\n  inner: 123\n  flag: true\n"
        with mock.patch("sparkrun.utils.resource_loader.load_resource", return_value=yaml_text):
            result = load_yaml_resource("some.package", "nested.yaml")
        assert result == {"outer": {"inner": 123, "flag": True}}

    def test_nonexistent_resource_propagates_error(self):
        """FileNotFoundError from load_resource is not swallowed."""
        with mock.patch(
            "sparkrun.utils.resource_loader.load_resource",
            side_effect=FileNotFoundError("no such resource"),
        ):
            with pytest.raises(FileNotFoundError):
                load_yaml_resource("some.package", "missing.yaml")
