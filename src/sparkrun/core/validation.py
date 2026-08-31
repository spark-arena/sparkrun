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
    """

    severity: str
    code: str
    summary: str
    fix: str = ""

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

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "summary": self.summary,
            "fix": self.fix,
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
    issues.extend(_safe("managed-comm-env", lambda: check_managed_comm_env(recipe)))
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


def validate_for_launch(
    recipe: Recipe,
    *,
    fail_on: str | None = None,
    config: SparkrunConfig | None = None,
    **kwargs,
) -> tuple[list[RecipeIssue], bool]:
    """Validate on a launch path: ``(issues_to_display, should_abort)``.

    The launch peer of :func:`validate_recipe`, shared by ``run``, the
    benchmark flow and ``proxy launch`` so the three cannot drift on what they
    print or what they refuse.  Two differences from ``recipe validate``:

    * **Suggestions are withheld.** They are advice for whoever *wrote* the
      recipe, and at launch you are usually running someone else's.  ``recipe
      validate`` is where an author (or registry CI) reads them.
    * The threshold defaults to ``validation.fail_on`` rather than to a flag,
      so a site can tighten every launch path at once.

    The two interact: a threshold strict enough to fail on suggestions also
    shows them (:func:`display_threshold`).  Refusing to launch over a finding
    that was never printed is the one combination that must not happen.

    ``kwargs`` is forwarded to :func:`validate_recipe` (``runtime``,
    ``cluster``, ``include_unmapped_keys``, …).
    """
    threshold = fail_on or (config.validation_fail_on if config is not None else DEFAULT_FAIL_ON)
    issues = validate_recipe(recipe, config=config, **kwargs)
    return at_or_above(issues, display_threshold(threshold)), should_fail(issues, threshold)


def _safe(name: str, check) -> list[RecipeIssue]:
    """Run *check*, returning ``[]` and a debug log if it raises."""
    try:
        return list(check())
    except Exception:
        logger.debug("Recipe validation check %r raised", name, exc_info=True)
        return []
