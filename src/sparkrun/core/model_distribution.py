"""Model transfer providers; manifests, validation and launch policy stay in core."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from inspect import signature
import logging
import math
from pathlib import PurePosixPath
from typing import Any, Callable, Protocol

from sparkrun.core.registration import enlist_registry_state, register_unique
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest, ValidatedModelBinding
from sparkrun.transports.session import HostSession

logger = logging.getLogger(__name__)
MODEL_DISTRIBUTION_API_VERSION = 1
MODEL_PULL_API_VERSION = 1
_PROVIDERS: dict[str, ModelDistributionProvider] = {}
enlist_registry_state(globals(), "_PROVIDERS")
_CONFIG: ContextVar[Any] = ContextVar("model_distribution_config", default=None)
_BINDINGS: ContextVar[dict[tuple[str, str], ValidatedModelBinding] | None] = ContextVar("validated_models", default=None)


@dataclass(frozen=True)
class ModelTarget:
    host: str
    cache_root: str
    transfer_host: str
    required_files: tuple[str, ...] = ()

    def __post_init__(self):
        if any(
            not name or name.startswith("-") or any(c.isspace() for c in name) or "\x00" in name for name in (self.host, self.transfer_host)
        ):
            raise ModelArtifactError("model target must identify a valid management and transfer host")
        if not PurePosixPath(self.cache_root).is_absolute() or "\x00" in self.cache_root:
            raise ModelArtifactError("model provider cache roots must be prepared absolute paths")


@dataclass(frozen=True)
class ModelCopyRequest:
    manifest: ModelArtifactManifest
    source_host: str | None
    source_cache_root: str
    targets: tuple[ModelTarget, ...]
    operation_id: str
    offline: bool = False
    dry_run: bool = False
    timeout: float = 7200
    config: Any = field(default=None, repr=False, compare=False)
    session: HostSession | None = field(default=None, repr=False, compare=False)
    cancelled: Callable[[], bool] = field(default=lambda: False, repr=False, compare=False)
    progress: Callable[[str, int], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.targets, tuple) or len({t.host for t in self.targets}) != len(self.targets):
            raise ModelArtifactError("model provider targets must be immutable and unique")
        if not PurePosixPath(self.source_cache_root).is_absolute() or "\x00" in self.source_cache_root:
            raise ModelArtifactError("model provider source must be a prepared absolute path")
        paths = {f.path for f in self.manifest.files}
        for target in self.targets:
            if (
                not isinstance(target.required_files, tuple)
                or len(set(target.required_files)) != len(target.required_files)
                or set(target.required_files) - paths
            ):
                raise ModelArtifactError("model transfer files must be unique members of the manifest")
        if not math.isfinite(self.timeout) or self.timeout < 0:
            raise ModelArtifactError("model transfer timeout must be finite and nonnegative")


@dataclass(frozen=True)
class ModelPullRequest(ModelCopyRequest):
    token: Callable[[], str | None] = field(default=lambda: None, repr=False, compare=False)


@dataclass(frozen=True)
class ModelTransferResult:
    manifest_id: str
    outcomes: dict[str, str]
    errors: dict[str, str] = field(default_factory=dict)


class ModelDistributionUnsupported(ModelArtifactError):
    """The request is unsupported, before any acquisition or cache mutation."""


class ModelDistributionFailed(ModelArtifactError):
    """A provider failed or violated its contract; automatic fallback is unsafe."""


class ModelDistributionProvider(Protocol):
    def copy(self, request: ModelCopyRequest) -> ModelTransferResult: ...


def register_model_distribution_provider(name: str, provider: ModelDistributionProvider) -> None:
    if not name or name in {"auto", "builtin"} or not callable(getattr(provider, "copy", None)):
        raise ValueError("model distribution provider needs a unique name and copy method")
    if getattr(provider, "api_version", MODEL_DISTRIBUTION_API_VERSION) != MODEL_DISTRIBUTION_API_VERSION:
        raise ValueError("unsupported model distribution provider API version")
    register_unique(_PROVIDERS, name, provider, description="model distribution provider")


def model_distribution_operation(function: Callable) -> Callable:
    call_signature = signature(function)

    @wraps(function)
    def scoped(*args, **kwargs):
        arguments = call_signature.bind(*args, **kwargs).arguments
        config = arguments.get("config", _CONFIG.get())
        if config is None:
            config = getattr(arguments.get("sctx"), "config", None)
        token = _CONFIG.set(config)
        bindings_token = _BINDINGS.set({}) if _BINDINGS.get() is None else None
        try:
            return function(*args, **kwargs)
        finally:
            _CONFIG.reset(token)
            if bindings_token is not None:
                _BINDINGS.reset(bindings_token)

    return scoped


def operation_config():
    return _CONFIG.get()


def _setting(config: Any, key: str, default: Any) -> Any:
    get = getattr(config, "get", None)
    return get(key, default) if callable(get) else default


def validation_policy(config=None) -> str:
    config = _CONFIG.get() if config is None else config
    # Every policy prepares manifests. Legacy permits missing-inventory
    # compatibility; providers always require the strict inventory contract.
    policy = _setting(config, "model_cache_validation", "legacy")
    provider = _setting(config, "model_distribution_provider", "auto")
    if not isinstance(provider, str):
        raise ModelArtifactError("model_distribution_provider must be auto, builtin or a provider name")
    if not isinstance(policy, str) or policy not in {"legacy", "metadata", "checksum"}:
        raise ModelArtifactError("model_cache_validation must be legacy, metadata or checksum")
    if policy == "legacy" and (provider not in {"auto", "builtin"} or (provider == "auto" and _PROVIDERS)):
        return "metadata"
    return policy


def record_binding(binding: ValidatedModelBinding) -> None:
    if not binding.report.complete or binding.report.manifest_id != binding.manifest.identity:
        raise ModelArtifactError("only validated model bindings can reach the runtime")
    bindings = _BINDINGS.get()
    if bindings is not None:
        key = binding.model, binding.host
        old = bindings.get(key)
        if old is not None and old.manifest.identity != binding.manifest.identity:
            raise ModelArtifactError("one host cannot bind the same model name to different artifacts")
        bindings[key] = binding


def prepared_model(model: str, host: str) -> ValidatedModelBinding | None:
    return (_BINDINGS.get() or {}).get((model, host))


def prepared_models() -> tuple[ValidatedModelBinding, ...]:
    return tuple((_BINDINGS.get() or {}).values())


def try_model_transfer(request: ModelCopyRequest, *, pull: bool = False) -> ModelTransferResult | None:
    """Return None for planned builtin work, never after partial provider work.

    A provider borrows the prepared session. Operation state is explicit in the
    request, including in callback threads; no ContextVar lookup is required.
    """
    selected = _setting(request.config, "model_distribution_provider", "auto")
    if not isinstance(selected, str):
        raise ModelArtifactError("model_distribution_provider must be auto, builtin or a provider name")
    if selected == "builtin" or (selected == "auto" and not _PROVIDERS):
        return None
    if selected == "auto":
        if len(_PROVIDERS) != 1:
            raise ModelArtifactError("multiple model providers enabled; set model_distribution_provider explicitly")
        selected = next(iter(_PROVIDERS))
    if selected not in _PROVIDERS:
        raise ModelArtifactError("model distribution provider is not enabled: " + str(selected))
    provider = _PROVIDERS[selected]
    method = getattr(provider, "pull" if pull else "copy", None)
    if not callable(method) and pull:
        return None  # Copy-only providers compose with builtin acquisition.
    if not callable(method):
        raise ModelDistributionFailed("model provider has no callable copy method")
    if request.offline and pull and getattr(provider, "supports_offline_pull", False) is not True:
        raise ModelDistributionUnsupported("selected model provider cannot acquire artifacts offline")
    if request.dry_run:
        return None  # Planning must not manufacture cache-readiness evidence.
    if request.cancelled():
        raise ModelDistributionFailed("model transfer cancelled")
    try:
        result = method(request)
    except ModelDistributionUnsupported as exc:
        fallback = _setting(request.config, "model_distribution_fallback", getattr(provider, "fallback_on_unsupported", False))
        if _setting(request.config, "model_distribution_provider", "auto") == "auto" and fallback is True:
            logger.warning("Model provider unavailable (%s); using builtin transfer", exc)
            return None
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise ModelDistributionFailed("model provider failed; no fallback after transfer starts") from exc
    if request.cancelled():
        raise ModelDistributionFailed("model transfer cancelled")
    hosts = {target.host for target in request.targets}
    if not isinstance(result, ModelTransferResult) or result.manifest_id != request.manifest.identity:
        raise ModelDistributionFailed("model provider returned a different artifact identity")
    if (
        not isinstance(result.outcomes, dict)
        or not isinstance(result.errors, dict)
        or any(not isinstance(state, str) for state in result.outcomes.values())
    ):
        raise ModelDistributionFailed("model provider returned malformed outcomes")
    if len(hosts) != len(request.targets) or set(result.outcomes) != hosts or set(result.errors) - hosts:
        raise ModelDistributionFailed("model provider must report exactly the requested targets")
    if any(state not in {"complete", "already_present", "failed"} for state in result.outcomes.values()):
        raise ModelDistributionFailed("model provider returned an invalid target state")
    if any(result.outcomes[h] != "failed" for h in result.errors):
        raise ModelDistributionFailed("model provider returned conflicting success/error evidence")
    if any(state == "failed" for state in result.outcomes.values()):
        raise ModelDistributionFailed(
            "model provider failed: "
            + "; ".join(h + ": " + result.errors.get(h, "transfer failed") for h, state in result.outcomes.items() if state == "failed")
        )
    return result
