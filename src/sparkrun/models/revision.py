"""Resolve a model's revision (branch, tag or nothing) to the commit it names.

The model-side peer of :mod:`sparkrun.containers.digest`, with the same two
sources:

* :func:`resolve_hub_commit`: what the revision points to **now**, from the
  Hub.
* :func:`resolve_cached_commit`: what the hosts' HuggingFace caches **already
  hold**, from each host's ``refs/<revision>``. Offline: it never leaves the
  cluster.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REF_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class RevisionResolutionError(RuntimeError):
    """A model revision could not be pinned; the message says why and what to do."""

    def __init__(self, message: str, *, missing: tuple[str, ...] = ()):
        super().__init__(message)
        self.missing = missing
        """Hosts whose cache lacks the model (empty for any other failure)."""


def is_commit(revision: str | None) -> bool:
    return bool(revision) and bool(_COMMIT_RE.match(str(revision)))


def is_hub_model(model: str) -> bool:
    """False for a pre-placed path, which has no revision to pin."""
    return bool(model) and not model.startswith(("/", "~", ".", "$"))


def _repo_id(model: str) -> str:
    from sparkrun.models.download import parse_gguf_model_spec

    return parse_gguf_model_spec(model)[0]


def resolve_hub_commit(model: str, revision: str | None = None, *, config=None) -> str:
    """The commit *revision* (default ``main``) currently names on the Hub."""
    if is_commit(revision):
        return str(revision)
    from sparkrun.models.hub import hub_metadata_call

    repo_id = _repo_id(model)

    def _lookup() -> str | None:
        from huggingface_hub import model_info

        try:
            return model_info(repo_id, revision=revision or None).sha
        except Exception as error:
            logger.debug("model_info(%s, %s) failed: %s", repo_id, revision, error)
            return None

    commit = hub_metadata_call("model revision", repo_id, revision, _lookup, config=config)
    if not is_commit(commit):
        raise RevisionResolutionError(
            "could not resolve %s@%s on the Hugging Face Hub (unreachable, gated without a token, or the metadata "
            "budget is spent); use --offline to pin the revision the hosts' caches already hold" % (repo_id, revision or "main")
        )
    return str(commit)


def _read_ref_script(model: str, revision: str, cache_dir: str | None) -> str:
    from sparkrun.core.config import resolve_hf_cache_home
    from sparkrun.models.download import model_cache_path
    from sparkrun.utils.shell import validate_interpolated_path

    path = validate_interpolated_path(model_cache_path(model, resolve_hf_cache_home(cache_dir)), field_name="model cache path")
    return 'cat "%s/refs/%s" 2>/dev/null || true\n' % (path, revision)


def resolve_cached_commit(
    model: str,
    revision: str | None,
    hosts: list[str],
    *,
    ssh_kwargs: dict | None = None,
    cache_dir: str | None = None,
) -> str:
    """The commit the hosts' caches hold for *revision* (default ``main``); they must agree.

    A host without the model is an error rather than skipped: an offline
    launch cannot download it there.
    """
    if is_commit(revision):
        return str(revision)
    ref = revision or "main"
    if not _REF_NAME_RE.match(ref) or ".." in ref:
        raise RevisionResolutionError("revision %r is not a plain branch or tag name" % ref)
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    results = run_remote_scripts_parallel(list(hosts), _read_ref_script(model, ref, cache_dir), timeout=15, **(ssh_kwargs or {}))
    found: dict[str, str] = {}
    for result in results:
        value = (result.stdout or "").strip()
        if result.success and is_commit(value):
            found[result.host] = value
    missing = [h for h in hosts if h not in found]
    if missing:
        raise RevisionResolutionError(
            "%s@%s is not in the Hugging Face cache on %s, so there is no local revision to pin"
            % (_repo_id(model), ref, ", ".join(missing)),
            missing=tuple(missing),
        )
    commits = set(found.values())
    if len(commits) != 1:
        detail = "; ".join("%s: %s" % (h, c[:12]) for h, c in sorted(found.items()))
        raise RevisionResolutionError("the hosts cache different commits of %s@%s (%s)" % (_repo_id(model), ref, detail))
    return commits.pop()


def resolve_local_cached_commit(model: str, revision: str | None, cache_dir: str | None = None) -> str | None:
    """The commit this machine's HuggingFace cache holds for *revision*, or ``None``."""
    from sparkrun.core.config import resolve_hf_cache_home
    from sparkrun.models.download import model_cache_path

    if is_commit(revision):
        return str(revision)
    ref_file = Path(model_cache_path(model, resolve_hf_cache_home(cache_dir))) / "refs" / (revision or "main")
    try:
        value = ref_file.read_text().strip()
    except OSError:
        return None
    return value if is_commit(value) else None
