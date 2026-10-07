# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: Apache-2.0

"""Optional image-copy providers; source policy and launch ordering stay in core."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from inspect import signature
import logging
import re
from typing import Any, Callable, Protocol

from sparkrun.core.registration import enlist_registry_state, register_unique

logger = logging.getLogger(__name__)
IMAGE_DISTRIBUTION_API_VERSION = 1
IMAGE_PULL_API_VERSION = 1
IMAGE_RUNTIME_API_VERSION = 1
_PROVIDERS: dict[str, ImageDistributionProvider] = {}
enlist_registry_state(globals(), "_PROVIDERS")
_CONFIG: ContextVar[Any] = ContextVar("image_distribution_config", default=None)
_RUNTIME_IMAGES: ContextVar[dict[tuple[str | None, str], str] | None] = ContextVar("image_runtime_references", default=None)
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


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
    runtime_images: dict[str, str] = field(default_factory=dict)
    """Verified, immutable Docker IDs keyed by management host.

    Optional for providers whose imported images retain their requested name.
    A registry pin may map to different IDs on different Docker stores.
    """


class ImageDistributionUnsupported(RuntimeError):
    """A provider cannot support this request before destination transfer starts."""


class ImageDistributionProvider(Protocol):
    """Optional fallback_on_unsupported=True opts into builtin fallback in auto.

    The user can override this default with container_distribution_fallback.
    Only ImageDistributionUnsupported (before transfer) is eligible.
    """

    def copy(self, request: ImageCopyRequest) -> ImageCopyResult: ...


def register_image_distribution_provider(name: str, provider: ImageDistributionProvider) -> None:
    if not name or name in {"auto", "builtin"} or not callable(getattr(provider, "copy", None)):
        raise ValueError("image distribution provider needs a unique name and copy method")
    register_unique(_PROVIDERS, name, provider, description="image distribution provider")


def image_distribution_operation(function: Callable) -> Callable:
    """Share config and verified runtime IDs across one launch/staging operation."""
    call_signature = signature(function)

    @wraps(function)
    def scoped(*args, **kwargs):
        arguments = call_signature.bind(*args, **kwargs).arguments
        config = arguments.get("config")
        if config is None:
            config = getattr(arguments.get("sctx"), "config", _CONFIG.get())
        token = _CONFIG.set(config)
        bindings_token = _RUNTIME_IMAGES.set({}) if _RUNTIME_IMAGES.get() is None else None
        try:
            return function(*args, **kwargs)
        finally:
            _CONFIG.reset(token)
            if bindings_token is not None:
                _RUNTIME_IMAGES.reset(bindings_token)

    return scoped


def _record_runtime_images(image: str, result: ImageCopyResult, *, dry_run: bool) -> None:
    if set(result.runtime_images) - set(result.outcomes):
        raise ValueError("Image provider returned a runtime image for an unrequested host")
    for host, reference in result.runtime_images.items():
        if not isinstance(reference, str) or not _IMAGE_ID.fullmatch(reference):
            raise ValueError("Image provider runtime references must be full immutable Docker IDs")
        if result.outcomes[host] not in {"complete", "already_present"} or host in result.errors:
            raise ValueError("Image provider returned a runtime image for a failed host")
    bindings = _RUNTIME_IMAGES.get()
    if bindings is not None and not dry_run:
        bindings.update({(host, image): ref for host, ref in result.runtime_images.items()})


def resolve_distributed_image(
    image: str,
    host: str | None,
    *,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    session: Any = None,
) -> str:
    """Resolve a verified registry pin to a host's installed immutable image.

    Ordinary tags, disabled providers, and dry runs remain unchanged. Optional
    provider lookup allows a later offline operation to recover a prior import.
    Providers must validate both their pin receipt and the resident Docker ID;
    a mutable alias by itself is not proof of a registry pin. An explicitly
    supplied session is borrowed and remains owned by the caller.
    """
    if dry_run or "@" not in image:
        return image
    bindings = _RUNTIME_IMAGES.get()
    if bindings is not None and (host, image) in bindings:
        return bindings[host, image]
    config = _CONFIG.get()
    selected = config.get("container_distribution_provider", "auto") if config is not None else "auto"
    if selected == "auto":
        if len(_PROVIDERS) != 1:
            return image
        selected = next(iter(_PROVIDERS))
    provider = _PROVIDERS.get(selected) if selected != "builtin" else None
    resolve = getattr(provider, "local_image", None)
    if not callable(resolve):
        return image
    from sparkrun.transports.session import SshHostSession

    ssh = ssh_kwargs or {}
    owns_session = session is None
    if owns_session:
        session = SshHostSession(ssh_user=ssh.get("ssh_user"), ssh_key=ssh.get("ssh_key"), ssh_options=ssh.get("ssh_options"))
    try:
        reference = resolve(
            ImageCopyRequest(
                image=image,
                source_host=host,
                targets=(),
                transfer_hosts=(),
                config=config,
                session=session,
                offline=True,
            )
        )
    finally:
        if owns_session and session is not None:
            session.close()
    if reference is None:
        return image
    if not isinstance(reference, str) or not _IMAGE_ID.fullmatch(reference):
        raise ValueError("Image provider resolved a pin to a non-immutable runtime reference")
    if bindings is not None:
        bindings[host, image] = reference
    return reference


def has_image_distribution_provider() -> bool:
    config = _CONFIG.get()
    selected = config.get("container_distribution_provider", "auto") if config is not None else "auto"
    return selected != "builtin" and (bool(_PROVIDERS) or selected != "auto")


def _fallback_allowed(config: Any, provider: ImageDistributionProvider) -> bool:
    default = getattr(provider, "fallback_on_unsupported", False) is True
    if config is None:
        return default
    return (
        config.get("container_distribution_provider", "auto") == "auto" and config.get("container_distribution_fallback", default) is True
    )


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
    session: Any = None,
) -> list[str] | None:
    """Return failed hosts, or None to use the builtin copy.

    A supplied host session is borrowed; only internally created sessions close.
    """
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

    # Existing callers use SSH; integrations can reuse their prepared transport.
    from sparkrun.transports.session import SshHostSession

    owns_session = session is None and not dry_run
    if owns_session:
        session = SshHostSession(ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options)
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
        if _fallback_allowed(config, _PROVIDERS[selected]):
            logger.warning("Image relay unavailable (%s); using Docker save/load fallback", error)
            return None
        logger.error("Image copy unsupported: %s", error)
        return list(targets)
    finally:
        if owns_session and session is not None:
            session.close()

    if not isinstance(result, ImageCopyResult) or set(result.outcomes) != set(targets):
        raise ValueError("Image-copy provider did not report every requested target exactly once")
    if any(status not in {"complete", "already_present", "failed", "cancelled"} for status in result.outcomes.values()):
        raise ValueError("Image-copy provider returned an invalid target state")
    if set(result.errors) - set(targets):
        raise ValueError("Image-copy provider reported errors for an unrequested target")
    _record_runtime_images(image, result, dry_run=dry_run)
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
    session: Any = None,
) -> list[str] | None:
    """Optional registry-to-target operation before the builtin source pull.

    None declines before payload transfer and preserves existing core policy.
    A provider lacking this additive capability continues through API v1 copy.
    Partial failure raises: an outer auto-delegated fallback must not select a
    different mutable-tag identity after some targets have already completed.
    A supplied host session is borrowed, including on failure.
    """
    if not targets:
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
    if not callable(pull) or (offline and not getattr(_PROVIDERS[selected], "supports_offline_pull", False)):
        return None
    addresses = targets if transfer_hosts is None else transfer_hosts
    if len(addresses) != len(targets) or len(set(targets)) != len(targets):
        raise ValueError("Image pull requires unique targets and aligned transfer addresses")
    from sparkrun.transports.session import SshHostSession

    owns_session = session is None and not dry_run
    if owns_session:
        session = SshHostSession(ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options)
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
        if _fallback_allowed(config, _PROVIDERS[selected]):
            logger.warning("Image pre-pull provider unavailable (%s); using builtin fallback", error)
            return None
        raise ImageDistributionFailed("Image pre-pull provider unsupported: " + str(error)) from error
    finally:
        if owns_session and session is not None:
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
    _record_runtime_images(image, result, dry_run=dry_run)
    return []
