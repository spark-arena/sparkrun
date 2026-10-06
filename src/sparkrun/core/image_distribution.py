# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: Apache-2.0

"""Optional image-copy providers; source policy and launch ordering stay in core."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from inspect import signature
import logging
from typing import Any, Callable, Protocol

from sparkrun.core.registration import enlist_registry_state, register_unique

logger = logging.getLogger(__name__)
IMAGE_DISTRIBUTION_API_VERSION = 1
IMAGE_PULL_API_VERSION = 1
_PROVIDERS: dict[str, ImageDistributionProvider] = {}
enlist_registry_state(globals(), "_PROVIDERS")
_CONFIG: ContextVar[Any] = ContextVar("image_distribution_config", default=None)


@dataclass(frozen=True)
class ImageCopyRequest:
    image: str
    source_host: str | None
    targets: tuple[str, ...]
    transfer_hosts: tuple[str, ...]
    ssh_user: str | None = None
    ssh_key: str | None = None
    ssh_options: tuple[str, ...] = ()
    timeout: float | None = None
    dry_run: bool = False
    offline: bool = False
    config: Any = field(default=None, compare=False, repr=False)
    session: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ImagePullRequest(ImageCopyRequest):
    """Optional pre-pull operation. The fetcher need not be a Docker target."""

    force_pull: bool = False


@dataclass(frozen=True)
class ImageCopyResult:
    """An outcome for every requested target; missing targets cannot succeed."""

    outcomes: dict[str, str]
    errors: dict[str, str] = field(default_factory=dict)


class ImageDistributionUnsupported(RuntimeError):
    """A provider cannot support this request before destination transfer starts."""


class ImageDistributionProvider(Protocol):
    def copy(self, request: ImageCopyRequest) -> ImageCopyResult: ...


def register_image_distribution_provider(name: str, provider: ImageDistributionProvider) -> None:
    if not name or name in {"auto", "builtin"} or not callable(getattr(provider, "copy", None)):
        raise ValueError("image distribution provider needs a unique name and copy method")
    register_unique(_PROVIDERS, name, provider, description="image distribution provider")


def image_distribution_operation(function: Callable) -> Callable:
    """Carry the caller's actual operation config without mutating recipe state."""
    call_signature = signature(function)

    @wraps(function)
    def scoped(*args, **kwargs):
        config = call_signature.bind(*args, **kwargs).arguments.get("config")
        token = _CONFIG.set(config)
        try:
            return function(*args, **kwargs)
        finally:
            _CONFIG.reset(token)

    return scoped


def has_image_distribution_provider() -> bool:
    config = _CONFIG.get()
    selected = config.get("container_distribution_provider", "auto") if config is not None else "auto"
    return selected != "builtin" and (bool(_PROVIDERS) or selected != "auto")


def try_image_copy(
    *,
    image: str,
    source_host: str | None,
    targets: list[str],
    transfer_hosts: list[str] | None,
    ssh_user: str | None,
    ssh_key: str | None,
    ssh_options: list[str] | None,
    timeout: float | None,
    dry_run: bool,
    offline: bool,
) -> list[str] | None:
    """Return failed management hosts, or None to use the built-in copy."""
    config = _CONFIG.get()
    selected = config.get("container_distribution_provider", "auto") if config is not None else "auto"
    if selected == "builtin" or (selected == "auto" and not _PROVIDERS):
        return None
    if selected == "auto":
        if len(_PROVIDERS) != 1:
            raise ValueError("Multiple image-copy providers enabled; set container_distribution_provider explicitly")
        selected = next(iter(_PROVIDERS))
    if selected not in _PROVIDERS:
        raise ValueError(f"Image distribution provider {selected!r} is not enabled")
    if not targets:
        return []
    addresses = targets if transfer_hosts is None else transfer_hosts
    if len(addresses) != len(targets) or len(set(targets)) != len(targets):
        raise ValueError("Image copy requires unique targets and aligned transfer addresses")

    # These leaf paths already represent SSH-connected hosts. Future callers may
    # supply another transport's session through an extended operation context.
    from sparkrun.transports.session import SshHostSession

    session = None if dry_run else SshHostSession(ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options)
    request = ImageCopyRequest(
        image=image,
        source_host=source_host,
        targets=tuple(targets),
        transfer_hosts=tuple(addresses),
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=tuple(ssh_options or ()),
        timeout=timeout,
        dry_run=dry_run,
        offline=offline,
        config=config,
        session=session,
    )
    try:
        result = _PROVIDERS[selected].copy(request)
    except ImageDistributionUnsupported as error:
        fallback = config.get("container_distribution_fallback", False) if config is not None else False
        if config is not None and fallback is True and config.get("container_distribution_provider", "auto") == "auto":
            logger.warning("Image relay unavailable (%s); using configured Docker save/load fallback", error)
            return None
        logger.error("Image copy unsupported: %s", error)
        return list(targets)
    finally:
        if session is not None:
            session.close()

    if not isinstance(result, ImageCopyResult) or set(result.outcomes) != set(targets):
        raise ValueError("Image-copy provider did not report every requested target exactly once")
    if any(status not in {"complete", "already_present", "failed", "cancelled"} for status in result.outcomes.values()):
        raise ValueError("Image-copy provider returned an invalid target state")
    if set(result.errors) - set(targets):
        raise ValueError("Image-copy provider reported errors for an unrequested target")
    return [host for host in targets if result.outcomes[host] in {"failed", "cancelled"} or host in result.errors]


class ImageDistributionFailed(RuntimeError):
    """A started pre-pull operation failed; do not retry a mutable tag elsewhere."""


def try_image_pull(
    *,
    image: str,
    source_host: str | None,
    targets: list[str],
    transfer_hosts: list[str] | None,
    ssh_user: str | None,
    ssh_key: str | None,
    ssh_options: list[str] | None,
    timeout: float | None,
    dry_run: bool,
    offline: bool,
    force_pull: bool,
) -> list[str] | None:
    """Optional registry-to-target operation before the builtin source pull.

    None declines before payload transfer and preserves existing core policy.
    A provider lacking this additive capability continues through API v1 copy.
    Partial failure raises: an outer auto-delegated fallback must not select a
    different mutable-tag identity after some targets have already completed.
    """
    if offline or not targets:
        return None
    config = _CONFIG.get()
    selected = config.get("container_distribution_provider", "auto") if config is not None else "auto"
    if selected == "builtin" or (selected == "auto" and not _PROVIDERS):
        return None
    if selected == "auto":
        if len(_PROVIDERS) != 1:
            raise ValueError("Multiple image-copy providers enabled; set container_distribution_provider explicitly")
        selected = next(iter(_PROVIDERS))
    if selected not in _PROVIDERS:
        raise ValueError(f"Image distribution provider {selected!r} is not enabled")
    pull = getattr(_PROVIDERS[selected], "pull", None)
    if not callable(pull):
        return None
    addresses = targets if transfer_hosts is None else transfer_hosts
    if len(addresses) != len(targets) or len(set(targets)) != len(targets):
        raise ValueError("Image pull requires unique targets and aligned transfer addresses")
    from sparkrun.transports.session import SshHostSession

    session = None if dry_run else SshHostSession(ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options)
    request = ImagePullRequest(
        image=image,
        source_host=source_host,
        targets=tuple(targets),
        transfer_hosts=tuple(addresses),
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=tuple(ssh_options or ()),
        timeout=timeout,
        dry_run=dry_run,
        offline=offline,
        force_pull=force_pull,
        config=config,
        session=session,
    )
    try:
        result = pull(request)
    except ImageDistributionUnsupported as error:
        if (
            config is not None
            and config.get("container_distribution_fallback", False) is True
            and config.get("container_distribution_provider", "auto") == "auto"
        ):
            logger.warning("Image pre-pull provider unavailable (%s); using configured builtin fallback", error)
            return None
        raise ImageDistributionFailed("Image pre-pull provider unsupported: " + str(error)) from error
    finally:
        if session is not None:
            session.close()
    if result is None:
        return None
    if not isinstance(result, ImageCopyResult) or set(result.outcomes) != set(targets):
        raise ValueError("Image-pull provider did not report every requested target exactly once")
    if any(state not in {"complete", "already_present", "failed", "cancelled"} for state in result.outcomes.values()):
        raise ValueError("Image-pull provider returned an invalid target state")
    if set(result.errors) - set(targets):
        raise ValueError("Image-pull provider reported errors for an unrequested target")
    failed = [host for host in targets if result.outcomes[host] in {"failed", "cancelled"} or host in result.errors]
    if failed:
        raise ImageDistributionFailed("Registry image distribution failed on: " + ", ".join(failed))
    return []
