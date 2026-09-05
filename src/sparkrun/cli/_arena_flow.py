"""Shared arena flow helpers: auth + submission_id preflight, post-benchmark upload.

Used by both ``sparkrun arena benchmark`` (the legacy alias path) and
``sparkrun benchmark perf --arena`` (the new flag-driven path) so the arena
behavior lives in exactly one place.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import click

from sparkrun.core.benchmark_profiles import ARENA_BENCHMARK_PROFILE  # noqa: F401 — re-exported for back-compat

logger = logging.getLogger(__name__)


def _is_interactive() -> bool:
    """``sys.stdin.isatty()``, behind a seam so both branches are testable.

    ``CliRunner`` replaces ``sys.stdin`` with a non-TTY stream for the duration
    of an invocation, so the interactive path is otherwise unreachable from a
    test — which would leave the branch that actually prompts unexercised.
    """
    return sys.stdin.isatty()


def _count_phrase(counts: "dict[str, int]") -> str:
    """ "3 warnings and 4 suggestions", omitting whichever level is absent."""
    from sparkrun.core.validation import SUGGESTION, WARNING

    parts = [
        "%d %s%s" % (counts[level], label, "" if counts[level] == 1 else "s")
        for level, label in ((WARNING, "warning"), (SUGGESTION, "suggestion"))
        if counts[level]
    ]
    return " and ".join(parts)


def validate_recipe_for_submission(recipe_name: str, *, ctx: click.Context, dry_run: bool = False) -> None:
    """Show the *full* validation report and confirm before an arena submission.

    The benchmark path already validates (``api._benchmark`` calls
    ``validate_for_launch``), so this is not the launch's safety check — it
    asks a different question.  ``validate_for_launch`` deliberately withholds
    suggestions and collapses deprecations, because at launch you are usually
    running someone else's recipe and the migration is not your work.  A
    submission inverts both premises: the recipe is *yours*, it is about to be
    published, and reproducibility for whoever reads it later is the whole
    point.  So this is the ``recipe validate`` report — every severity, nothing
    collapsed — shown at the one moment the author can still act on it.

    Gated to the arena paths for that reason and no other.  An ordinary
    ``sparkrun benchmark`` publishes nothing, and printing suggestions there
    would undo the rule the launch paths share.

    Errors abort without prompting: the launch would refuse anyway, and
    "continue?" is not a real question when the answer cannot be yes.  Warnings
    and suggestions prompt, because both are things a submitter can reasonably
    decide to accept.  A clean recipe says nothing at all — a report with no
    findings is noise between the user and their benchmark.
    """
    from sparkrun.core.validation import ERROR, SUGGESTION, WARNING, validate_recipe
    from sparkrun.utils.cli_formatters import format_validation_report
    from ._common import _get_config_and_registry, _get_context, _load_recipe

    sctx = _get_context(ctx)
    config, _ = _get_config_and_registry()
    recipe, _path, _mgr = _load_recipe(config, recipe_name)

    issues = validate_recipe(recipe, config=config, v=sctx.variables)
    if not issues:
        return

    counts = {level: sum(1 for i in issues if i.severity == level) for level in (ERROR, WARNING, SUGGESTION)}
    click.echo(
        format_validation_report(
            recipe.qualified_name,
            issues,
            title="Validating recipe before submission",
        )
    )
    click.echo()

    if counts[ERROR]:
        click.echo(
            "The recipe has %d error%s and cannot be launched, so it cannot be submitted."
            % (counts[ERROR], "" if counts[ERROR] == 1 else "s"),
            err=True,
        )
        sys.exit(1)

    click.echo(
        "The recipe contains %s. We suggest that you address all of these issues before submitting a "
        "benchmark; problematic recipes may not be published to Spark Arena." % _count_phrase(counts)
    )

    if dry_run:
        click.echo("[dry-run] Not prompting; nothing will be submitted.")
        click.echo()
        return

    # Not a security gate (that is the hook trust prompt, which refuses without
    # a TTY). This is quality advice on a path that people script, so a
    # non-interactive run proceeds and says so rather than aborting a
    # submission that worked yesterday.
    if not _is_interactive():
        click.echo("Not running interactively — continuing without confirmation.")
        click.echo()
        return

    if not click.confirm("Continue with the benchmark and submission?", default=False):
        click.echo("Aborted.")
        sys.exit(1)
    click.echo()


def preflight_arena(
    *,
    local_test: bool,
    ctx: click.Context,
    recipe_name: str,
    dry_run: bool = False,
) -> tuple[str, str | None]:
    """Authenticate (unless ``local_test``), validate the recipe, mint a submission id.

    Returns ``(submission_id, profile)`` — profile is ``None`` in local-test
    mode (caller chooses what to use) or the arena's pinned profile name.

    Validation runs *after* auth so a user who is not logged in is told the
    thing that blocks them outright before being asked to review advice, and
    so the report — and its prompt — sit directly above the launch they gate.
    It runs in local-test mode too: that path simulates the upload, and
    rehearsing a submission without its checks would defeat the rehearsal.

    *recipe_name* is required rather than optional so a future arena entry
    point cannot bypass the gate by forgetting to pass it.
    """
    from sparkrun.arena.upload import generate_submission_id

    if local_test:
        click.echo("[local-test] Skipping authentication — upload will be simulated.")
        click.echo()
        validate_recipe_for_submission(recipe_name, ctx=ctx, dry_run=dry_run)
        submission_id = generate_submission_id()
        return submission_id, None

    from sparkrun.arena.auth import load_refresh_token, exchange_token

    refresh_token = load_refresh_token()
    if not refresh_token:
        click.echo("Error: Not logged in. Run 'sparkrun arena login' first.", err=True)
        sys.exit(1)

    try:
        exchange_token(refresh_token)
    except RuntimeError as e:
        click.echo("Error: Authentication failed: %s" % e, err=True)
        click.echo("Run 'sparkrun arena login' to re-authenticate.", err=True)
        sys.exit(1)

    click.echo("Authentication verified.")
    click.echo()

    validate_recipe_for_submission(recipe_name, ctx=ctx, dry_run=dry_run)

    submission_id = generate_submission_id()
    return submission_id, ARENA_BENCHMARK_PROFILE


def finalize_arena(
    *,
    ctx: click.Context,
    bench_result: Any,
    submission_id: str,
    local_test: bool,
    dry_run: bool,
) -> None:
    """Persist arena extras and (unless local_test/dry_run) upload results.

    Idempotent: if upload was already performed for this submission_id, the
    upload module short-circuits.
    """
    if dry_run:
        click.echo("[dry-run] Would upload results to Spark Arena")
        return

    from sparkrun.arena.auth import load_refresh_token
    from sparkrun.arena.upload import upload_benchmark_results
    from ._common import _get_context

    # Gather recipe/metadata/runtime info (works with or without launch_result)
    recipe = bench_result.launch_result.recipe if bench_result.launch_result else bench_result.recipe
    overrides = bench_result.launch_result.overrides if bench_result.launch_result else (bench_result.overrides or {})
    # Published off this machine — the host set is recorded as stable
    # pseudonyms so a leaderboard submission never carries the user's LAN
    # addresses or internal hostnames.
    metadata = bench_result.generate_metadata(redact_hosts=True)
    effective_recipe = recipe.export(
        overrides=overrides,
        container_image=metadata["recipe"]["container"],
    )
    benchmark_csv = bench_result.results["csv"]

    if local_test:
        from pprint import pformat

        click.echo("-" * 40)
        click.echo("Effective Recipe Export:")
        click.echo("-" * 40)
        click.echo(effective_recipe)
        click.echo("-" * 40)
        click.echo("Metadata:")
        click.echo("-" * 40)
        click.echo(pformat(metadata))
        click.echo("-" * 40)

    # --- Write files to cache directory ---
    sctx = _get_context(ctx)
    cache_dir = sctx.config.cache_dir / "benchmarks" / submission_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    recipe_path = cache_dir / "recipe.yaml"
    csv_path = cache_dir / "benchmark.csv"
    meta_path = cache_dir / "metadata.json"

    recipe_path.write_text(effective_recipe)
    csv_path.write_text(benchmark_csv)
    meta_path.write_text(json.dumps(metadata, indent=2))

    upload_files = [
        (recipe_path, "recipes"),
        (csv_path, "logs"),
        (meta_path, "metadata"),
    ]

    # Persist arena-specific extras in state for resume support
    persist_arena_extras(ctx, bench_result, submission_id, effective_recipe, metadata)

    if local_test:
        click.echo()
        click.echo("[local-test] Benchmark files written to: %s" % cache_dir)
        for fpath, folder in upload_files:
            click.echo("  %s -> %s/" % (fpath.name, folder))
        click.echo("[local-test] Skipping actual upload.")
        return

    # --- Upload results ---
    refresh_token = load_refresh_token()

    click.echo()
    click.echo("Uploading results to Spark Arena...")

    try:
        success, sid = upload_benchmark_results(
            refresh_token=refresh_token,
            upload_files=upload_files,
            submission_id=submission_id,
        )
    except RuntimeError as e:
        click.echo("Upload failed: %s" % e, err=True)
        sys.exit(1)

    if success:
        click.echo("Results uploaded successfully (submission: %s)" % sid)
    else:
        click.echo("Some files failed to upload (submission: %s)" % sid, err=True)
        sys.exit(1)


def persist_arena_extras(ctx, bench_result: Any, submission_id: str, effective_recipe: str, metadata: dict) -> None:
    """Persist arena-specific fields into BenchmarkRunState.extras if a state exists.

    Extracted from ``_arena.py:_persist_arena_extras`` so both arena entry
    paths share the same persistence logic.
    """
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from ._common import _get_context

    sctx = _get_context(ctx)
    config = sctx.config
    cache_dir = str(config.cache_dir) if config else None

    # cluster_id and framework are needed to derive benchmark_id; they live on bench_result
    cluster_id = bench_result.cluster_id
    if not cluster_id:
        return

    fw = bench_result.framework
    bench_args = getattr(bench_result, "benchmark_args", None)
    if fw is None or bench_args is None:
        return

    if not config:
        return

    benchmarks_dir = config.cache_dir / "benchmarks"
    if not benchmarks_dir.exists():
        return

    for candidate_dir in benchmarks_dir.iterdir():
        if not candidate_dir.is_dir():
            continue
        state = BenchmarkRunState.load(candidate_dir.name, cache_dir)
        if state is None:
            continue
        if state.cluster_id != cluster_id:
            continue
        # Found the matching state; persist extras (do not overwrite existing submission_id)
        if "submission_id" not in state.extras:
            state.extras["submission_id"] = submission_id
        state.extras["effective_recipe_text"] = effective_recipe
        state.extras["metadata_json"] = metadata
        state.save(cache_dir)
        logger.debug("Persisted arena extras into benchmark state %s", state.benchmark_id)
        break


__all__ = ["preflight_arena", "finalize_arena", "persist_arena_extras", "ARENA_BENCHMARK_PROFILE"]
