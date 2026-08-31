"""Recipe validation — the single aggregator behind ``recipe validate``.

Three severities, and the rule that separates them is one question:

    *If this recipe runs on a cluster that isn't the author's, does it break
    or behave differently?*

**Errors** — sparkrun cannot do what the recipe asks.  A missing required
field, a runtime that rejects the configuration, a named builder or executor
that does not resolve.  No setting makes proceeding correct, so these always
fail ``recipe validate`` and always abort a launch.

**Warnings** — *yes*, it breaks or behaves differently.  NCCL pinned to one
machine's device names, a bind mount only the author has, a ``defaults:`` key
the runtime drops so the engine silently substitutes its own value.  Each is a
supported escape hatch, which is why they cannot be errors; each is also
exactly the shape that works on its author's cluster and misbehaves quietly on
everyone else's.

There is a second axis on which a recipe "breaks somewhere else", and it is
**time** rather than place: a *deprecated* feature works identically on every
cluster today and stops working on a future sparkrun.  Read literally, the
question above answers "no" and would file every deprecation as a suggestion —
which would mean ``sparkrun run`` never mentions it, since suggestions are
withheld at launch.  So the rule is stated with both axes: a warning is
something that breaks or behaves differently on **another cluster, or on a
later sparkrun**.  A deprecation is a warning.  This does not add a fourth
severity — ``--fail-on`` still ranks three, and ``--strict`` still means "fail
on anything that is not merely advice".

**Suggestions** — *no*.  It works as written; it just gives up something.  A
serve flag hardcoded in ``command:`` serves precisely what it says — you just
cannot override it.  A malformed ``metadata`` label costs an estimate, equally
on every host.  These are for recipe authors and registry CI, not for whoever
is launching: ``sparkrun run`` does not print them at all.

Only errors are fatal by default.  ``--strict`` (i.e. ``--fail-on warning``)
is the CI posture; ``--fail-on suggestion`` is the author polishing their own
recipe.  See :func:`should_fail`.

The catalogue below is deliberately conservative.  A check earns its place by
naming something sparkrun itself reads or computes — where the cost of the
recipe overriding it is silent, not loud.  Anything fuzzier (style, taste,
"most recipes do X") belongs in docs, not here: a finding nobody can act on
teaches people to ignore the ones they can.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.base import RuntimePlugin

logger = logging.getLogger(__name__)

ERROR = "error"
WARNING = "warning"
SUGGESTION = "suggestion"

#: Severities from most to least severe.  Rank is used for sorting the report,
#: for the ``--fail-on`` threshold, and for deciding what a launch prints.
SEVERITIES = (ERROR, WARNING, SUGGESTION)
_RANK = {name: i for i, name in enumerate(SEVERITIES)}

#: ``--fail-on`` / ``validation.fail_on`` values.  ``"none"`` reports without
#: ever failing (scripting that wants the JSON and not the exit code).
NEVER = "none"
FAIL_ON_CHOICES = (*SEVERITIES, NEVER)

#: The default threshold everywhere: only what sparkrun *cannot honor* is
#: fatal.  Anything stricter is opt-in, per invocation or via
#: ``validation.fail_on``.
DEFAULT_FAIL_ON = ERROR


def rank(severity: str) -> int:
    """Position in :data:`SEVERITIES`; unknown severities sort last."""
    return _RANK.get(severity, len(SEVERITIES))


def should_fail(issues: Iterable[RecipeIssue], fail_on: str = DEFAULT_FAIL_ON) -> bool:
    """True when any finding is at or above the *fail_on* threshold.

    ``fail_on=ERROR`` (the default) fails only on things sparkrun cannot do;
    ``WARNING`` adds the portability class (this is what ``--strict`` means);
    ``SUGGESTION`` fails on everything; :data:`NEVER` never fails.
    """
    if fail_on == NEVER:
        return False
    limit = rank(fail_on)
    return any(rank(i.severity) <= limit for i in issues)


def display_threshold(fail_on: str = DEFAULT_FAIL_ON) -> str:
    """The least-severe level a *launch* should print, given *fail_on*.

    Launches show errors and warnings but not suggestions — a hardcoded serve
    flag is advice for whoever wrote the recipe, not for whoever is running it.
    The one exception is a threshold that would *fail* on suggestions: never
    refuse to launch over a finding that was never shown.

    :data:`NEVER` is checked before the rank comparison. It is not a *stricter*
    level than ``suggestion`` even though it sorts after one (``rank`` puts
    unknown values last), and widening the output for the threshold that fails
    at nothing would be exactly backwards.
    """
    if fail_on == NEVER:
        return WARNING
    return SUGGESTION if rank(fail_on) >= rank(SUGGESTION) else WARNING


def at_or_above(issues: Iterable[RecipeIssue], severity: str) -> list[RecipeIssue]:
    """Findings at least as severe as *severity*, in the order given."""
    limit = rank(severity)
    return [i for i in issues if rank(i.severity) <= limit]


@dataclass(frozen=True)
class RecipeIssue:
    """One validation finding.

    The diagnosis and the remediation are separate fields because they are
    read at different moments: *what is wrong* decides whether you care, *what
    to do* only matters once you do.  Run together they make a paragraph the
    eye slides off — and these messages are long, because a check that names a
    defect without naming the fix just relocates the puzzle.  ``recipe
    validate`` renders them as separate blocks; the compact call sites keep
    using :attr:`message`, which is the two joined.

    Args:
        severity: :data:`ERROR` (sparkrun cannot honor this), :data:`WARNING`
            (it will, but the recipe breaks or behaves differently off its
            author's cluster) or :data:`SUGGESTION` (it works as written and
            merely gives something up).  See the module docstring for the rule.
        code: Stable kebab-case identifier for the check, so tooling can
            filter or suppress by kind without matching on prose.
        summary: What is wrong, and why it matters.
        fix: What to do about it.  Empty when the summary already contains the
            only possible action (e.g. "model is required").
        deprecation: This names a feature that works now and is scheduled to
            stop working.  Declared rather than read off the ``code`` prefix,
            because an out-of-tree runtime naming its own check still needs to
            be able to say so.  Only :func:`summarize_deprecations` reads it —
            it is a *display* distinction, not a fourth severity.
    """

    severity: str
    code: str
    summary: str
    fix: str = ""
    deprecation: bool = False

    @property
    def is_error(self) -> bool:
        return self.severity == ERROR

    @property
    def is_suggestion(self) -> bool:
        return self.severity == SUGGESTION

    @property
    def message(self) -> str:
        """Summary and fix as one string — the single-line rendering."""
        return "%s %s" % (self.summary, self.fix) if self.fix else self.summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "summary": self.summary,
            "fix": self.fix,
            "deprecation": self.deprecation,
        }

    def __str__(self) -> str:
        return self.message


def errors(issues: Iterable[RecipeIssue]) -> list[RecipeIssue]:
    return [i for i in issues if i.severity == ERROR]


def warnings(issues: Iterable[RecipeIssue]) -> list[RecipeIssue]:
    return [i for i in issues if i.severity == WARNING]


def suggestions(issues: Iterable[RecipeIssue]) -> list[RecipeIssue]:
    return [i for i in issues if i.severity == SUGGESTION]


def coerce_issues(findings: Iterable[Any], code: str, *, default_severity: str = SUGGESTION) -> list[RecipeIssue]:
    """Normalize a ``validate_recipe()`` hook's return into :class:`RecipeIssue`.

    The hook's signature is ``list[str] | list[RecipeIssue]``, and the two
    forms mean different things on purpose:

    * A **plain string** is *undeclared* severity and becomes
      *default_severity* — the **least** severe level.  Every hook returned
      strings before severity existed, and their wording ranges from a hard
      incompatibility ("modular-max is single-node only") to a style note
      ("command template is recommended"), so reading severity out of the
      prose would be guesswork.  Guessing *up* is the dangerous direction —
      it would newly fail launches that work — so the unknown lands at the
      bottom.  Out-of-tree plugins built against the older base class keep
      working, and their findings stay non-fatal exactly as before.
    * A :class:`RecipeIssue` is a plugin that has *said* which it is.  Its
      ``code`` is kept when it set one, so a runtime can name its own check
      rather than being lumped under ``runtime-field``.
    """
    out: list[RecipeIssue] = []
    for finding in findings:
        if isinstance(finding, RecipeIssue):
            # Re-code only; summary and fix stay split so the renderer can
            # still lay them out as separate blocks.
            out.append(finding if finding.code else RecipeIssue(finding.severity, code, finding.summary, finding.fix))
        else:
            out.append(RecipeIssue(default_severity, code, str(finding)))
    return out


# --------------------------------------------------------------------------
# Hardcoded-serve-flag check
# --------------------------------------------------------------------------

#: Config keys sparkrun *itself* reads, with why — the only keys for which a
#: literal in the ``command:`` template is worth a warning.
#:
#: A recipe is free (and encouraged) to hardcode engine flags sparkrun has no
#: opinion about; that is what ``command:`` is for.  These are different: the
#: value feeds a sparkrun decision made *outside* the serve command, and a
#: literal is invisible to it.  ``--kv-cache-dtype auto`` written into the
#: template is how a VRAM estimate silently computes at the wrong width
#: (issue #248); a hardcoded ``--served-model-name`` is how a benchmark asks
#: for the wrong model id and gets HTTP 404 for the whole sweep (#257); a
#: hardcoded ``--tensor-parallel-size`` is worse still, because placement sizes
#: the cluster from the config chain and would pick the wrong node count.
SPARKRUN_READ_CONFIG_KEYS: Mapping[str, str] = {
    "tensor_parallel": "cluster placement and world size",
    "pipeline_parallel": "cluster placement and world size",
    "data_parallel": "cluster placement and world size",
    "max_model_len": "VRAM estimation",
    "gpu_memory_utilization": "VRAM estimation and the memory budget",
    "kv_cache_dtype": "KV cache sizing",
    "served_model_name": "benchmark request targets, proxy routing and container labels",
    "port": "health checks, workload identity and proxy discovery",
}

#: Extra spellings to look for beyond the runtime's own flag map, since the
#: same key is spelled differently across engines and a recipe may use a long
#: form where the map holds the short one (vLLM's map says ``-tp``; real
#: recipes almost always write ``--tensor-parallel-size``).
_FLAG_ALIASES: Mapping[str, tuple[str, ...]] = {
    "tensor_parallel": ("--tensor-parallel-size", "-tp", "--tp-size", "--tp", "--tp_size"),
    "pipeline_parallel": ("--pipeline-parallel-size", "-pp", "--pp-size", "--pp_size"),
    "data_parallel": ("--data-parallel-size", "-dp", "--dp-size", "--dp_size"),
    # Deliberately no ``-c``: llama.cpp's short form for --ctx-size, but it
    # also appears as ``taskset -c`` and ``bash -c`` in real command templates.
    "max_model_len": ("--max-model-len", "--max_seq_len", "--context-length", "--ctx-size"),
    "gpu_memory_utilization": ("--gpu-memory-utilization", "--mem-fraction-static"),
    "kv_cache_dtype": ("--kv-cache-dtype", "--kv-cache-dtype-str"),
    "served_model_name": ("--served-model-name", "--model-name", "--alias"),
    "port": ("--port",),
}


def _command_flag_value(command: str, flag: str) -> str | None:
    """Return the token following *flag* in *command*, or ``None`` if absent.

    Handles both ``--flag value`` and ``--flag=value``.  Returns the empty
    string for a flag that appears with no following token (a bare boolean),
    which callers treat as "present with a literal".

    Line-continuation backslashes are dropped first: every real recipe wraps
    its serve command, and a trailing ``\\`` would otherwise be read as the
    value of the flag preceding it.
    """
    tokens = [t for t in command.split() if t != "\\"]
    for idx, token in enumerate(tokens):
        if token == flag:
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            # A following token that is itself a flag means this one is bare.
            return "" if nxt.startswith("-") else nxt
        if token.startswith(flag + "="):
            return token[len(flag) + 1 :]
    return None


def check_hardcoded_serve_flags(recipe: Recipe, runtime: RuntimePlugin | None) -> list[RecipeIssue]:
    """Warn when ``command:`` pins a value sparkrun needs to read.

    Only fires when *all* of the following hold, which is what keeps it quiet
    on the recipes that are doing the right thing:

    * the key is in :data:`SPARKRUN_READ_CONFIG_KEYS`,
    * a flag spelling for it appears in ``command:`` with a **literal** value
      (a ``{placeholder}`` is the documented, working pattern and is fine),
    * the key is absent from ``defaults:`` — a recipe that declares the value
      *and* spells the flag is already visible to the config chain.
    """
    command = recipe.command or ""
    if not command:
        return []

    flag_map: Mapping[str, str] = {}
    if runtime is not None:
        try:
            flag_map = runtime.serve_flag_map() or {}
        except Exception:  # pragma: no cover - defensive, mirrors known_config_keys
            logger.debug("Runtime %r serve_flag_map raised", getattr(runtime, "runtime_name", "?"), exc_info=True)

    found: list[RecipeIssue] = []
    for key, why in SPARKRUN_READ_CONFIG_KEYS.items():
        if key in recipe.defaults:
            continue
        spellings = set(_FLAG_ALIASES.get(key, ()))
        mapped = flag_map.get(key)
        if mapped:
            spellings.add(mapped)
        for flag in sorted(spellings):
            value = _command_flag_value(command, flag)
            if value is None or "{" in value:
                continue
            found.append(
                RecipeIssue(
                    # A suggestion, not a warning: the rendered command serves
                    # exactly what it says, identically on every cluster. What
                    # is lost is *configurability* — the value can't be seen or
                    # overridden through the config chain — not correctness.
                    SUGGESTION,
                    "hardcoded-serve-flag",
                    "command: hardcodes '%s %s' but does not declare defaults.%s. sparkrun reads %s from the "
                    "config chain, so a literal here is invisible to it (and to `-o %s=...`)." % (flag, value or "", key, why, key),
                    "Move the value to defaults.%s and reference it as {%s} in the command." % (key, key),
                )
            )
            break

    return found


# --------------------------------------------------------------------------
# Sparkrun-managed communication env
# --------------------------------------------------------------------------


def check_managed_comm_env(recipe: Recipe) -> list[RecipeIssue]:
    """Warn when ``env:`` overrides values sparkrun detects per cluster.

    ``merge_env(nccl_env, env)`` puts ``recipe.env`` last, so the recipe wins
    outright — the detected HCA list, GID index and interface names are
    replaced by whatever the recipe hardcoded.  On the cluster the recipe was
    written on that is invisible; anywhere else it pins NCCL to device names
    that may not exist, and the failure surfaces as a collective hang rather
    than as anything pointing back at the recipe.
    """
    from sparkrun.orchestration.infiniband import MANAGED_COMM_ENV_KEYS

    overridden = sorted(k for k in (recipe.env or {}) if k in MANAGED_COMM_ENV_KEYS)
    if not overridden:
        return []
    return [
        RecipeIssue(
            WARNING,
            "managed-comm-env",
            "env: sets %s, which sparkrun detects per cluster from the live InfiniBand/CX7 probe. The recipe's "
            "values win over the detected ones, so this pins the recipe to one machine's device naming and will "
            "misconfigure NCCL on any host whose adapters are named differently." % ", ".join(overridden),
            "Remove them and let detection fill them in, unless you are deliberately overriding it for one specific cluster.",
        )
    ]


# --------------------------------------------------------------------------
# Sparkrun-managed cache env
# --------------------------------------------------------------------------


def check_managed_cache_env(recipe: Recipe, runtime: RuntimePlugin | None) -> list[RecipeIssue]:
    """Report ``env:`` entries that collide with sparkrun's own cache wiring.

    The cache env is set at two tiers that sit on **opposite sides** of
    ``recipe.env`` in ``merge_env`` (``RuntimePlugin._run_solo`` and
    ``_cluster_ops``), so the same-looking mistake has two opposite outcomes
    and needs two different sentences:

    * ``get_extra_env()`` is merged **last** — it carries ``HF_HOME`` /
      ``HF_HUB_CACHE``, which have to beat the ``XDG_CACHE_HOME`` catch-all or
      the model cache silently relocates off its own mount.  A recipe setting
      those is therefore **overridden**: the line is dead, and reads as though
      it works.
    * the runtime-cache tier is merged **first** — ``XDG_CACHE_HOME`` and the
      runtime's declared compile/autotune paths.  A recipe setting those
      **wins**, which moves the caches off the mount sparkrun prepared and
      persists across launches; since containers are ``--rm``, the visible
      symptom is a full recompile every launch and nothing naming the recipe.

    Both are warnings and neither is an error: pointing these somewhere else
    is a legitimate thing to want, and the first is harmless beyond the
    misdirection.  What is not legitimate is not being told which one you did.
    """
    env = recipe.env or {}
    if not env or runtime is None:
        return []

    from sparkrun.core.runtime_cache import XDG_CACHE_ENV

    def _hook(name: str, **kwargs) -> Iterable[str]:
        # Defensive for the same reason ``known_config_keys`` is: an
        # out-of-tree runtime built against an older base class, or one whose
        # hook raises, must cost this diagnostic and nothing more.
        try:
            return (getattr(runtime, name)(**kwargs) or {}).keys()
        except Exception:
            logger.debug("Runtime %r %s raised", getattr(runtime, "runtime_name", "?"), name, exc_info=True)
            return ()

    found: list[RecipeIssue] = []

    overridden = sorted(k for k in env if k in set(_hook("get_extra_env")))
    if overridden:
        found.append(
            RecipeIssue(
                WARNING,
                "overridden-cache-env",
                "env: sets %s, which the '%s' runtime sets after the recipe's env is merged. The recipe's values "
                "are discarded — these point at sparkrun's HuggingFace cache mount, and have to win over the "
                "XDG_CACHE_HOME catch-all or the model cache relocates and the weights re-download every launch. "
                "The recipe reads as though it redirects the cache and does not."
                % (", ".join(overridden), getattr(runtime, "runtime_name", recipe.runtime)),
                "Remove them. To put the cache somewhere else, point sparkrun's own cache at it "
                "(`cache_dir` in config.yaml, or the cluster's cache settings) rather than the container's env.",
            )
        )

    cache_keys = {XDG_CACHE_ENV, *_hook("runtime_cache_paths")}
    wins = sorted(k for k in env if k in cache_keys and k not in set(_hook("get_extra_env")))
    if wins:
        found.append(
            RecipeIssue(
                WARNING,
                "managed-cache-env",
                "env: sets %s, which sparkrun points at the persistent runtime-cache mount it prepares on each "
                "host. This tier is merged *below* recipe env, so the recipe wins and the compile/autotune caches "
                "land somewhere sparkrun neither creates nor persists. Containers run with --rm, so the effect is "
                "a full torch.compile / Triton / autotune rebuild on every launch, with nothing pointing back at "
                "the recipe." % ", ".join(wins),
                "Remove them and let sparkrun place the cache. To keep a recipe-specific tree, set "
                "`runtime_cache:` in the recipe instead — it keys the host directory without moving the mount.",
            )
        )

    return found


# --------------------------------------------------------------------------
# Unknown top-level keys
# --------------------------------------------------------------------------

#: ``runtime_config`` keys something actually reads *by name*, with where.
#: These are legitimately written at the top level — that is the v1 spelling,
#: and the sweep exists to keep it working — so they are not "unknown".
#:
#: There is no registry to derive this from: each is a ``runtime_config.get()``
#: at its own call site, so a new one has to be added here too.  The list is
#: short and the cost of forgetting is one soft suggestion, but the cost of
#: *not* having it is a guaranteed false positive on every v1 recipe ever
#: published — and a finding that fires on correct recipes is how a report
#: teaches people to skim it.
_CONSUMED_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "build_args",  # builders/eugr.py
        "mods",  # core/recipe.py migrates it to the top-level field
        "dashboard",  # runtimes/vllm_ray.py
        "mmproj",  # core/launcher.py
    }
)


def check_unknown_top_level_keys(recipe: Recipe, runtime: RuntimePlugin | None) -> list[RecipeIssue]:
    """Report top-level keys swept into ``runtime_config``.

    An unrecognized top-level key is not rejected: ``Recipe.__init__`` absorbs
    it into ``runtime_config``, a v1 compatibility path that keeps ``mods`` and
    ``build_args`` working where those recipes wrote them.  But nothing
    consumes ``runtime_config`` generically — it is read *by name* by the few
    features that want it (``build_args``, ``mods``, ``dashboard``,
    ``mmproj``) — so anything else lands there and is never looked at again.

    Nothing reports this today.  ``report_unmapped_config_keys`` is the
    closest thing and it reads ``defaults`` and ``-o`` overrides, which is
    precisely the set an absorbed top-level key is *not* in.  So the two
    shapes below are silent in a way the rest of the config chain is not:

    * a key the runtime **does** understand as serve configuration, written at
      the top level instead of under ``defaults:``.  That is a warning: the
      recipe states a value, the engine runs with its own instead, and the
      rendered command shows nothing missing.  ``max_model_len: 8192`` at the
      top level is the shape — the same typo class as #276, one level up.
    * anything else — a typo (``defualts:``), a stale key, a field from a
      newer sparkrun.  A suggestion, because a key this build does not know is
      routinely a *newer* recipe rather than a broken one (the reasoning
      ``report_unmapped_config_keys`` already follows), and because a runtime
      or plugin may legitimately read it by name.
    """
    from sparkrun.core.recipe import _KNOWN_KEYS
    from sparkrun.core.recipe_items import registered_recipe_items

    raw = getattr(recipe, "_raw", None) or {}
    plugin_keys = {registration.key for registration in registered_recipe_items()}
    # Only keys the *sweep* absorbed. An explicit ``runtime_config:`` mapping
    # is a deliberate statement about a runtime that reads it by name.
    explicit = set((raw.get("runtime_config") or {}) if isinstance(raw.get("runtime_config"), Mapping) else ())
    ignored = _KNOWN_KEYS | plugin_keys | explicit | _CONSUMED_RUNTIME_CONFIG_KEYS
    absorbed = sorted(str(k) for k in raw if k not in ignored)
    if not absorbed:
        return []

    known: set[str] = set()
    if runtime is not None:
        try:
            from sparkrun.runtimes.base import BASE_CONSUMED_CONFIG_KEYS

            declared = runtime.known_config_keys()
            if declared is not None:
                known = set(declared) | set(BASE_CONSUMED_CONFIG_KEYS)
        except Exception:
            logger.debug("Runtime %r known_config_keys raised", getattr(runtime, "runtime_name", "?"), exc_info=True)

    misplaced = [k for k in absorbed if k in known]
    unknown = [k for k in absorbed if k not in known]

    found: list[RecipeIssue] = []
    if misplaced:
        found.append(
            RecipeIssue(
                WARNING,
                "misplaced-config-key",
                "These are serve-configuration keys the '%s' runtime understands, but they are written at the top "
                "level of the recipe: %s. Only `defaults:` feeds the config chain, so at the top level they are "
                "absorbed into runtime_config, reach nothing, and the engine runs with its own values — with the "
                "rendered command showing nothing missing." % (getattr(runtime, "runtime_name", recipe.runtime), ", ".join(misplaced)),
                "Move them under `defaults:`.",
            )
        )
    if unknown:
        found.append(
            RecipeIssue(
                SUGGESTION,
                "unknown-top-level-key",
                "Top-level key(s) sparkrun does not recognize: %s. They are absorbed into runtime_config, which is "
                "read by name by the few features that use it (build_args, mods, dashboard, mmproj) and by nothing "
                "generically — so unless a runtime or plugin asks for one of these by name, they have no effect." % ", ".join(unknown),
                "Check the spelling against RECIPES.md; move serve flags under `defaults:`. If a key is deliberate "
                "and read by name, declare it under `runtime_config:` so it reads as intentional.",
            )
        )
    return found


# --------------------------------------------------------------------------
# Bind-mount portability
# --------------------------------------------------------------------------

#: Host path prefixes that are present (or sparkrun-managed) on every target,
#: so mounting them says nothing about where the recipe can run.
_PORTABLE_MOUNT_PREFIXES = (
    "/dev/",
    "/sys/",
    "/proc/",
    "/run/",
    "/var/run/",
    "/tmp/",
    "/etc/localtime",
    "/etc/timezone",
    "/usr/share/zoneinfo",
)


def _is_portable_mount_source(path: str) -> bool:
    if not path.startswith("/"):
        # A named docker volume or a relative path — not a host-path claim.
        return True
    if path.startswith(_PORTABLE_MOUNT_PREFIXES) or path in ("/dev", "/sys", "/proc", "/run", "/tmp"):
        return True
    from sparkrun.core.config import DEFAULT_CACHE_DIR, DEFAULT_HF_CACHE_DIR

    managed: list[str] = []
    for root in (DEFAULT_CACHE_DIR, DEFAULT_HF_CACHE_DIR):
        managed.extend((str(root), str(root).rstrip("/") + "/"))
    return path.startswith(tuple(managed))


def check_mount_portability(recipe: Recipe) -> list[RecipeIssue]:
    """Warn about absolute host paths in ``executor_config.volumes``.

    The *static* peer of the launch-time probe in
    :func:`sparkrun.core.launcher._verify_mount_sources`, and the division of
    labour matters: the probe is authoritative but needs a cluster to ask, so
    it can only speak at launch and only about the hosts being launched on.
    This runs with no hosts at all — in ``recipe validate``, in CI, in a
    registry lint — and answers the question the probe cannot: *would this
    recipe be portable off the machine it was written on?*  A site-specific
    path passes the probe on its author's cluster every time.

    Warning rather than error because the path is a legitimate escape hatch on
    a cluster where it does exist; the launch-time check is what fails
    (``mounts.missing_source``, default ``fail``) when it doesn't.

    Issue #262 is the canonical shape: an ``@spark-arena`` recipe shipped its
    submitter's ``/home/nvidia/...`` (the DGX Spark factory-default login) in
    three ``volumes:`` entries, so on any host with a different login user
    Docker created all three **root-owned** while sparkrun's rootless default
    ran the workload as the SSH user — surfacing as ``PermissionError: [Errno
    13] ... '/cache/inductor'`` from inside the container, with nothing
    pointing back at the recipe.
    """
    raw = (recipe.executor_config or {}).get("volumes")
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, Mapping):
        sources: list[str] = [str(k) for k in raw]
    elif isinstance(raw, (list, tuple)):
        sources = [str(spec).split(":", 1)[0] for spec in raw if spec]
    else:
        return []

    offenders = sorted({p for p in sources if not _is_portable_mount_source(p)})
    if not offenders:
        return []
    return [
        RecipeIssue(
            WARNING,
            "non-portable-mount",
            "executor_config.volumes binds host path(s) %s that look specific to the machine the recipe was "
            "authored on (a home directory, a site-local mount). Docker creates a missing bind source as an "
            "empty root-owned directory instead of failing, and sparkrun runs the container as the SSH user by "
            "default — so elsewhere this serves without the content that was meant to be there, or dies with a "
            "permission error from inside the container." % ", ".join("'%s'" % p for p in offenders),
            "Package the content as a `mods:` entry — a mod is copied into the container and run before the "
            "serve command, so it travels with the recipe through its registry and needs no host path. "
            "Otherwise bake it into the container image, or move it under a sparkrun-managed cache path. "
            "(`sparkrun run` verifies these on the target hosts and refuses to launch when they are missing; "
            "see `mounts.missing_source`.)",
        )
    ]


# --------------------------------------------------------------------------
# Deprecated recipe features
# --------------------------------------------------------------------------

#: Runtimes that still launch and are on their way out, with the migration.
#: The deprecation is announced today only by a ``logger.warning`` from
#: ``prepare()`` — i.e. mid-launch, after distribution.
_DEPRECATED_RUNTIMES: Mapping[str, str] = {
    "eugr-vllm": "Set `runtime: vllm-ray` with `builder: eugr` instead — the same code path under a supported name.",
}

#: eugr ``build_args`` entries the builder still accepts and then ignores.
_DEPRECATED_BUILD_ARGS: Mapping[str, str] = {
    "--tf5": "it has been a no-op since the image tag it selected was retired",
}

#: The v1 topology surface, and the ``min_nodes``/``max_nodes`` that replaces
#: each spelling.  Read straight out of ``Recipe.__init__``, which is the only
#: place these three have any effect — every one of them is applied *by*
#: rewriting the node range, so the migration is exact rather than approximate.
#: ``mode: cluster`` is the one that does not fold into the range: it only
#: makes ``sparkrun run`` print a warning when it lands on one host, whereas
#: ``min_nodes: 2`` is enforced by validation and by placement.  So the
#: deprecated spelling is also the weaker one, which is worth saying.
_DEPRECATED_TOPOLOGY: Mapping[str, str] = {
    "mode: solo": "`max_nodes: 1`",
    "mode: cluster": "`min_nodes: 2`",
    "mode: auto": "nothing — it is the default; delete the key",
    "solo_only": "`max_nodes: 1`",
    "cluster_only": "`min_nodes: 2`",
}


#: What a deprecated topology key that is *set to a falsy value* becomes.  It
#: constrains nothing, so ``solo_only: false`` does not migrate to
#: ``max_nodes: 1`` — that would be the opposite instruction, and telling
#: someone to pin a node range they explicitly declined is worse than saying
#: nothing.
_TOPOLOGY_NOOP = "nothing — it constrains nothing as written; delete the key"


def _declared_topology(recipe: Recipe) -> list[tuple[str, str]]:
    """``(spelling, migration)`` for each v1 topology key the recipe *wrote*.

    ``_raw`` again, and for a sharper reason than elsewhere: ``recipe.mode`` is
    **derived**.  ``Recipe.__init__`` rewrites ``auto`` to ``cluster`` or
    ``solo`` from the node range, so every recipe has a non-default ``mode``
    by the time anything can read it — including the ones that correctly use
    ``min_nodes``/``max_nodes`` and never mentioned ``mode`` at all.  Reporting
    off the parsed value would fire on exactly the recipes this advises people
    to write.
    """
    raw = getattr(recipe, "_raw", None) or {}
    declared: list[tuple[str, str]] = []
    if "mode" in raw:
        spelling = "mode: %s" % raw.get("mode")
        # An unrecognized mode is already a ``validate_structure`` error; here
        # it is still a deprecated key, just one with no specific migration.
        declared.append((spelling, _DEPRECATED_TOPOLOGY.get(spelling, "`min_nodes` / `max_nodes`")))
    for key in ("solo_only", "cluster_only"):
        if key in raw:
            declared.append((key, _DEPRECATED_TOPOLOGY[key] if raw.get(key) else _TOPOLOGY_NOOP))
    return declared


def _raw_escaped_default_keys(recipe: Recipe) -> list[str]:
    """``defaults:`` keys written with the v1 doubled-brace escape.

    Read off ``recipe._raw`` rather than ``recipe.defaults``, and that is not
    an optimization: :func:`~sparkrun.core.recipe._resolve_brace_escapes`
    collapses the escape **in place** at load, so by the time any validation
    runs ``{{`` is already gone from the parsed defaults and only the raw YAML
    still carries the evidence.  ``command:`` needs no such treatment — it is
    masked inside ``render_command`` and the stored template is untouched.
    """
    from sparkrun.utils.text import uses_brace_escapes

    raw = (getattr(recipe, "_raw", None) or {}).get("defaults")
    if not isinstance(raw, Mapping):
        return []
    return sorted(str(k) for k, val in raw.items() if isinstance(val, str) and uses_brace_escapes(val))


def check_deprecated_features(recipe: Recipe) -> list[RecipeIssue]:
    """Report recipe features that work today and are scheduled to stop.

    This is the static peer of deprecation notices that otherwise only ever
    reach a *launch*, and two of them could not reach ``recipe validate`` at
    all: the doubled-brace notice for ``command:`` lives inside
    ``Recipe.render_command``, which validation never calls, and the
    ``eugr-vllm`` notice lives inside that runtime's ``prepare()``, which runs
    after image distribution.  So the command that exists to tell you what is
    wrong with a recipe was silent about both, while ``sparkrun run`` was not
    — the gap this closes.

    Warnings, per the module docstring's second axis: a deprecation behaves
    identically everywhere today and breaks on a later sparkrun.  At launch
    they are collapsed to one line by :func:`summarize_deprecations`, since
    the migration is the author's work rather than the runner's.
    """
    from sparkrun.utils.text import uses_brace_escapes

    found: list[RecipeIssue] = []
    is_v1 = recipe.recipe_version == "1"

    if is_v1:
        found.append(
            RecipeIssue(
                WARNING,
                "deprecated-recipe-format",
                "recipe_version: '1' is the legacy eugr format. sparkrun migrates it on load — which is why it "
                "still runs — and the migration is inference: it picks the runtime from the command template and "
                "sets `builder: %s` from the presence of `build_args`/`mods`." % (recipe.builder or "eugr"),
                "Convert to the v2 format (see RECIPES.md), declaring `runtime:` and `builder:` explicitly. Nothing "
                "about the resulting deployment changes; it stops depending on the migration.",
                deprecation=True,
            )
        )
    else:
        # Gated on *not* v1 rather than on the escape alone: doubled braces are
        # the correct spelling in a v1 recipe, and reporting them there would
        # be advice to write something the format does not support.
        if recipe.command and uses_brace_escapes(recipe.command):
            found.append(
                RecipeIssue(
                    WARNING,
                    "deprecated-brace-escape",
                    "command: uses the v1 doubled-brace escape ('{{' / '}}') in a recipe_version '%s' recipe — "
                    "usually a template pasted from a v1 recipe. sparkrun detects the convention from '{{' and "
                    "collapses it, so the runtime receives valid JSON today." % recipe.recipe_version,
                    "Write literal braces plainly instead (--flag '%s'); a {placeholder} nested inside JSON still "
                    "resolves. The escape will not be supported by v3 recipes. Note the convention is read off the "
                    "template as a whole, so a template mixing both spellings collapses the '}}' that merely closes "
                    "a nested plain-JSON object." % '{"key": "value"}',
                    deprecation=True,
                )
            )
        escaped = _raw_escaped_default_keys(recipe)
        if escaped:
            found.append(
                RecipeIssue(
                    WARNING,
                    "deprecated-brace-escape",
                    "These defaults use the v1 doubled-brace escape ('{{' / '}}') in a recipe_version '%s' recipe: "
                    "%s. sparkrun collapses them on load, so the value reaches the engine correctly today."
                    % (recipe.recipe_version, ", ".join("defaults.%s" % k for k in escaped)),
                    "Write literal braces plainly instead. The escape will not be supported by v3 recipes.",
                    deprecation=True,
                )
            )

    migration = _DEPRECATED_RUNTIMES.get(recipe.runtime)
    if migration:
        found.append(
            RecipeIssue(
                WARNING,
                "deprecated-runtime",
                "runtime: '%s' is deprecated. It still launches, and says so only once the launch is already "
                "underway — its deprecation notice is logged from prepare(), after image distribution." % recipe.runtime,
                migration,
                deprecation=True,
            )
        )

    # Topology. Reported for v1 as well as v2 — unlike the brace escape, this
    # is not a spelling v1 requires, it is a spelling v1 *introduced*, and the
    # replacement has always worked in both. A v1 recipe already carries the
    # format finding above, so this adds the one thing that one does not say:
    # which keys, and what each becomes.
    topology = _declared_topology(recipe)
    if topology:
        one = len(topology) == 1
        found.append(
            RecipeIssue(
                WARNING,
                "deprecated-topology",
                "%s %s retained for backward compatibility with v1 recipes and deprecated for v2 and later. "
                "%s applied by rewriting min_nodes/max_nodes, so %s already an indirect spelling of the node "
                "range — except `mode: cluster`, which only makes `sparkrun run` print a warning if the launch "
                "lands on one host, where `min_nodes: 2` is enforced by validation and by placement."
                % (
                    ", ".join("`%s`" % spelling for spelling, _ in topology),
                    "is" if one else "are",
                    "It is" if one else "They are",
                    "it is" if one else "they are",
                ),
                "Use min_nodes / max_nodes instead — they are explicit, and they compose with the rest of the "
                "recipe surface. Here: %s." % "; ".join("`%s` → %s" % (spelling, becomes) for spelling, becomes in topology),
                deprecation=True,
            )
        )

    build_args = (recipe.runtime_config or {}).get("build_args")
    if isinstance(build_args, (list, tuple)):
        # Match the bare flag and its ``--flag=value`` spelling; the builder
        # accepts either and ignores both.
        dead = sorted(
            {
                flag
                for flag, _why in _DEPRECATED_BUILD_ARGS.items()
                for arg in build_args
                if str(arg) == flag or str(arg).startswith(flag + "=")
            }
        )
        for flag in dead:
            found.append(
                RecipeIssue(
                    WARNING,
                    "deprecated-build-arg",
                    "build_args contains '%s', which the eugr builder accepts and then ignores — %s. The image is "
                    "built as though the flag were absent." % (flag, _DEPRECATED_BUILD_ARGS[flag]),
                    "Remove it. If the recipe depends on the image it used to select, pin that image in `container:`.",
                    deprecation=True,
                )
            )

    return found


# --------------------------------------------------------------------------
# Inferred builder
# --------------------------------------------------------------------------


def check_implicit_builder(recipe: Recipe) -> list[RecipeIssue]:
    """Note a ``builder:`` that was inferred rather than declared.

    ``_resolve_eugr_signals`` sets ``builder: eugr`` from ``build_args`` or an
    eugr ``container:`` reference, and ``_resolve_v1_migration`` does the same
    for v1 recipes.  Both are heuristics over recipe *content*, so what gets
    built is decided by rules that live in sparkrun and version with it — not
    by anything the recipe says.  That is a fine back-compatibility path and
    thin ice to depend on: the recipe reads as though it has no builder, and a
    later sparkrun is free to weigh the signals differently.

    **A first-party image is not reported.**  The concern is "sparkrun guessed
    something about an artifact it does not control, and that guess can change
    underneath the recipe".  For an image the project publishes itself
    (:data:`~sparkrun.core.recipe.FIRST_PARTY_CONTAINER_PREFIX`) that concern
    does not apply — sparkrun owns both the image and the rule, so keeping them
    consistent is its job, not the recipe author's.  This is the bulk of the
    real catalogue: of 47 registry recipes with an inferred builder, 40 were
    inferred from a first-party ``container:`` and only 7 from ``build_args``.
    Reporting all 47 would have made the finding mostly noise, and noise is how
    a report teaches people to skim it.

    **The finding names the signal it actually found**, rather than listing the
    catalogue ("``build_args``/``mods``, or a container reference").  A list
    reads as a guess, and a reader matching it against their own recipe lands
    on the wrong entry — ``mods`` is the trap, since it is *not* a signal (it
    was, before it became part of the v2 spec) yet appears in plenty of recipes
    that were flagged for their container.

    A suggestion, not a warning: today it resolves identically on every
    cluster and on every sparkrun that has shipped, and declaring the builder
    changes nothing about the launch.  What it buys is that the recipe stops
    depending on inference.  v1 recipes are excluded — the
    ``deprecated-recipe-format`` finding already names the inferred builder,
    and reporting the same inference twice teaches people to skim.
    """
    if recipe.recipe_version == "1":
        return []
    declared = str((getattr(recipe, "_raw", None) or {}).get("builder") or "")
    if declared or not recipe.builder:
        return []

    from sparkrun.core.recipe import FIRST_PARTY_CONTAINER_PREFIX

    container = str(recipe.container or "").strip()
    if container.startswith(FIRST_PARTY_CONTAINER_PREFIX):
        return []

    if (recipe.runtime_config or {}).get("build_args"):
        signal = "its `build_args`"
    else:
        signal = "its `container:` reference"

    return [
        RecipeIssue(
            SUGGESTION,
            "implicit-builder",
            "builder: is not declared, but sparkrun inferred '%s' from %s. The image is therefore built by a rule "
            "that lives in sparkrun and versions with it, rather than by anything the recipe states — and unlike a "
            "first-party `%s` image, this one is not sparkrun's to keep in step." % (recipe.builder, signal, FIRST_PARTY_CONTAINER_PREFIX),
            "Declare `builder: %s` explicitly. It resolves to exactly what is happening now, and pins it." % recipe.builder,
        )
    ]


# --------------------------------------------------------------------------
# Builder / executor resolution
# --------------------------------------------------------------------------


def check_builder(recipe: Recipe, v: Variables | None = None) -> tuple[list[RecipeIssue], Any]:
    """Resolve ``recipe.builder`` and validate its fields.

    Returns ``(issues, builder_or_None)``.  A named-but-unresolvable builder
    is an **error**: unlike an unknown serve flag (which degrades to the
    engine's own default), a builder that does not run means the image or
    environment the recipe asked for is never built.  The workload then
    launches against whatever the ``container:`` reference happens to pull —
    which is why ``builder: ursuciprian`` on an ``@spark-arena`` recipe looked
    like a clean launch.  Same reasoning as
    :class:`~sparkrun.builders.base.BuilderUnavailableError`, extended to the
    unknown case.
    """
    if not recipe.builder:
        return [], None

    from sparkrun.builders.base import BuilderUnavailableError
    from sparkrun.core.bootstrap import get_builder

    try:
        builder = get_builder(recipe.builder, v)
    except BuilderUnavailableError as e:
        return [RecipeIssue(ERROR, "builder-disabled", str(e))], None
    except ValueError as e:
        return [RecipeIssue(ERROR, "builder-unknown", str(e))], None

    # Same string-or-RecipeIssue contract as the runtime hook — see
    # :func:`coerce_issues`.
    try:
        return coerce_issues(builder.validate_recipe(recipe), "builder-field"), builder
    except Exception:  # pragma: no cover - a builder hook must not break validation
        logger.debug("Builder %r validate_recipe raised", recipe.builder, exc_info=True)
        return [], builder


def check_executor(
    recipe: Recipe,
    *,
    runtime: RuntimePlugin | None = None,
    cluster: ClusterDefinition | None = None,
    config: SparkrunConfig | None = None,
    v: Variables | None = None,
) -> list[RecipeIssue]:
    """Report an ``executor:`` selector that will not resolve at launch.

    Resolution already fails closed at launch
    (:class:`~sparkrun.orchestration.executor.ExecutorUnavailableError`); this
    surfaces the same verdict from ``recipe validate``, and distinguishes
    gated-off from unknown exactly as the launch path does.
    """
    from sparkrun.orchestration.executor import ExecutorUnavailableError, resolve_executor_name

    try:
        resolve_executor_name(recipe=recipe, runtime=runtime, cluster=cluster, config=config, v=v)
    except ExecutorUnavailableError as e:
        return [RecipeIssue(ERROR, "executor-unavailable", str(e))]
    except Exception:  # pragma: no cover - never let a diagnostic break validation
        logger.debug("Executor resolution raised during validation", exc_info=True)
    return []


# --------------------------------------------------------------------------
# Aggregator
# --------------------------------------------------------------------------


def validate_recipe(
    recipe: Recipe,
    *,
    runtime: RuntimePlugin | None = None,
    overrides: dict[str, Any] | None = None,
    cluster: ClusterDefinition | None = None,
    config: SparkrunConfig | None = None,
    v: Variables | None = None,
    include_unmapped_keys: bool = True,
) -> list[RecipeIssue]:
    """Run every check against *recipe* and return the findings.

    *runtime* may be passed when the caller has already resolved it (the
    launch path has); otherwise it is resolved here so an unknown ``runtime:``
    is reported as an error and the runtime-dependent checks are skipped
    rather than guessed at.

    *include_unmapped_keys* folds in
    :func:`~sparkrun.core.launcher.report_unmapped_config_keys`.  The launch
    path turns it off because ``launch_inference`` runs that check itself,
    later — after the platform default tier has contributed its flags, and
    with ``-o`` already split into serve vs executor keys.

    Every check is run through :func:`_safe`, so a check that trips over an
    unexpected recipe shape costs its own finding and nothing else.  The same
    discipline ``report_unmapped_config_keys`` already follows, and it matters
    more here because ``sparkrun run`` now *aborts* on an error: a diagnostic
    that raises must not be able to block a launch it was only meant to
    describe.

    Most severe first, stable within each level.  Callers decide what is fatal
    (:func:`should_fail`) and what to display (:func:`display_threshold`); this
    always returns everything it found.
    """
    from sparkrun.core.bootstrap import get_runtime

    issues: list[RecipeIssue] = []
    issues.extend(_safe("recipe-field", lambda: [RecipeIssue(ERROR, "recipe-field", m) for m in recipe.validate_structure()]))
    # Metadata problems cost an *estimate*, and cost it identically on every
    # host — the estimator drops the claim and placement falls back to
    # capacity-only. Nothing about the served deployment changes, so this is a
    # suggestion rather than a portability warning.
    issues.extend(_safe("recipe-metadata", lambda: [RecipeIssue(SUGGESTION, "recipe-metadata", m) for m in recipe.validate_metadata()]))

    if runtime is None:
        try:
            runtime = get_runtime(recipe.runtime, v)
        except ValueError:
            issues.append(RecipeIssue(ERROR, "unknown-runtime", "Unknown runtime: %s" % recipe.runtime))

    # A hook may return plain strings (undeclared → warning) or RecipeIssues
    # (severity the plugin chose). See :func:`coerce_issues`.
    if runtime is not None:
        issues.extend(_safe("runtime-field", lambda: coerce_issues(runtime.validate_recipe(recipe), "runtime-field")))

    issues.extend(_safe("builder", lambda: check_builder(recipe, v)[0]))
    issues.extend(_safe("executor", lambda: check_executor(recipe, runtime=runtime, cluster=cluster, config=config, v=v)))
    issues.extend(_safe("deprecated-feature", lambda: check_deprecated_features(recipe)))
    issues.extend(_safe("implicit-builder", lambda: check_implicit_builder(recipe)))
    issues.extend(_safe("managed-comm-env", lambda: check_managed_comm_env(recipe)))
    issues.extend(_safe("managed-cache-env", lambda: check_managed_cache_env(recipe, runtime)))
    issues.extend(_safe("unknown-top-level-key", lambda: check_unknown_top_level_keys(recipe, runtime)))
    issues.extend(_safe("non-portable-mount", lambda: check_mount_portability(recipe)))
    issues.extend(_safe("hardcoded-serve-flag", lambda: check_hardcoded_serve_flags(recipe, runtime)))

    if runtime is not None and include_unmapped_keys:
        from sparkrun.core.launcher import report_unmapped_config_keys

        issues.extend(
            _safe(
                "unmapped-config-key",
                lambda: [
                    RecipeIssue(WARNING, "unmapped-config-key", m)
                    for m in report_unmapped_config_keys(recipe, runtime, overrides, log=False)
                ],
            )
        )

    # Most severe first, stable within a level (a level's checks run in a fixed
    # order, so the report is reproducible — which matters for CI diffs).
    return sorted(issues, key=lambda i: rank(i.severity))


def summarize_deprecations(issues: Iterable[RecipeIssue], recipe_ref: str) -> list[RecipeIssue]:
    """Collapse every ``deprecation`` finding into a single pointer line.

    A deprecation is worth *mentioning* at launch — you should know you are
    running something with a shelf life — but the detail is the recipe
    author's work, not the runner's, and at launch you are usually running
    someone else's recipe.  Printed in full, three deprecations are three
    paragraphs of migration instructions between you and your logs.

    So the launch path says one line and names where the rest is.  Severity is
    untouched (see :func:`validate_for_launch`): this is what gets *displayed*,
    never what gets *decided*, so ``--strict`` still fails on a deprecation it
    only summarized.  The summary keeps the highest severity among the
    findings it replaces, so it can never sort above or below them.

    Order is preserved: the summary lands where the first deprecation was.
    """
    issues = list(issues)
    deprecations = [i for i in issues if i.deprecation]
    if len(deprecations) < 1:
        return issues

    codes = sorted({i.code for i in deprecations})
    summary = RecipeIssue(
        min((i.severity for i in deprecations), key=rank),
        "deprecated-feature",
        "Recipe '%s' uses %d deprecated recipe feature%s (%s) that work now and may not in a future version of "
        "sparkrun." % (recipe_ref, len(deprecations), "" if len(deprecations) == 1 else "s", ", ".join(codes)),
        "Run `sparkrun recipe validate %s` for the details and the migration for each." % recipe_ref,
        deprecation=True,
    )

    out: list[RecipeIssue] = []
    for issue in issues:
        if not issue.deprecation:
            out.append(issue)
        elif issue is deprecations[0]:
            out.append(summary)
    return out


def validate_for_launch(
    recipe: Recipe,
    *,
    fail_on: str | None = None,
    config: SparkrunConfig | None = None,
    recipe_ref: str | None = None,
    **kwargs,
) -> tuple[list[RecipeIssue], bool]:
    """Validate on a launch path: ``(issues_to_display, should_abort)``.

    The launch peer of :func:`validate_recipe`, shared by ``run``, the
    benchmark flow and ``proxy launch`` so the three cannot drift on what they
    print or what they refuse.  Three differences from ``recipe validate``:

    * **Suggestions are withheld.** They are advice for whoever *wrote* the
      recipe, and at launch you are usually running someone else's.  ``recipe
      validate`` is where an author (or registry CI) reads them.
    * **Deprecations are collapsed to one line** naming ``recipe validate``
      (:func:`summarize_deprecations`), for the same reason and with the same
      remedy.
    * The threshold defaults to ``validation.fail_on`` rather than to a flag,
      so a site can tighten every launch path at once.

    The first and third interact: a threshold strict enough to fail on
    suggestions also shows them (:func:`display_threshold`).  Refusing to
    launch over a finding that was never printed is the one combination that
    must not happen — which is also why the collapse runs **after**
    :func:`should_fail`, on the display list only.  A summarized deprecation
    is still a deprecation as far as ``--strict`` is concerned; it is simply
    described in one line rather than five.

    *recipe_ref* is the reference to echo back — whatever the user typed, so
    the suggested ``recipe validate`` command is one they can paste.  Defaults
    to the recipe's qualified name.

    ``kwargs`` is forwarded to :func:`validate_recipe` (``runtime``,
    ``cluster``, ``include_unmapped_keys``, …).
    """
    threshold = fail_on or (config.validation_fail_on if config is not None else DEFAULT_FAIL_ON)
    issues = validate_recipe(recipe, config=config, **kwargs)
    failed = should_fail(issues, threshold)
    display = summarize_deprecations(at_or_above(issues, display_threshold(threshold)), recipe_ref or recipe.qualified_name)
    return display, failed


def _safe(name: str, check) -> list[RecipeIssue]:
    """Run *check*, returning ``[]` and a debug log if it raises."""
    try:
        return list(check())
    except Exception:
        logger.debug("Recipe validation check %r raised", name, exc_info=True)
        return []
