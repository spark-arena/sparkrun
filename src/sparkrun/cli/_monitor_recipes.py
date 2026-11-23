"""Recipe selection screens for ``sparkrun cluster monitor``.

The picker (``RecipePickerScreen``) browses the recipe catalog; picking one
runs the pre-flight interstitial (``RuntimeChecksScreen``: a progress bar with
Cancel while the scheduler placement and per-host memory fit are computed) and
then opens the parameter modal
(``RunParamsScreen``), whose Run button re-runs the same check with the final
options and only then hands a ``RecipeLaunch`` back to the app, which owns the
actual ``sparkrun.api.run`` call. Import direction is
one-way: ``_monitor_tui`` imports this module, never the reverse, and nothing
here touches the running workload view.
"""

from __future__ import annotations

import itertools
import shlex
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ItemGrid, Vertical
from textual.content import Content
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Checkbox, Footer, Input, OptionList, ProgressBar, Select, Static
from textual.widgets.option_list import Option

from sparkrun.api import (
    HostsUnreachable,
    InsufficientCapacity,
    RecipeSummary,
    RunOptions,
    catalog_cluster_capacity,
    plan,
    search_recipes,
    untrusted_hooks,
)
from sparkrun.api._resolve import resolve_recipe  # private-module import; CLI precedent: cli/_run.py resolve_operation_target
from sparkrun.core.application_profile import get_application_profile
from sparkrun.core.bootstrap import get_runtime
from sparkrun.core.recipe import is_local_model_path, recipe_matches_query
from sparkrun.core.resolve import apply_recipe_overrides
from sparkrun.models.fit import check_fit
from sparkrun.utils.runtime_display import RUNTIME_DISPLAY

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe


class BackgroundWork:
    """Blocking calls still running on worker threads, by description.

    Textual's thread workers run on the event loop's default executor, and a
    cancelled worker only stops *waiting*: its thread keeps going until the call
    returns. ``asyncio.run`` then joins those threads after the screen is
    restored, so quitting during a launch, a first catalog load (which may clone
    registries) or a scheduler check left the shell blocked with nothing on it.
    The app reads this registry to confirm a quit that would wait on a launch,
    and to say what it is waiting for once the screen is gone. The calls are not
    abandoned on daemon threads instead: a launch interrupted halfway leaves a
    half-started workload, the catalog load writes the registry cache, and
    planning prepares provider transports.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids = itertools.count()
        self._entries: dict[int, tuple[str, bool]] = {}

    @contextmanager
    def track(self, description: str, *, blocks_quit: bool = False) -> Iterator[Callable[[str], None]]:
        """Register *description* while the block runs; yields a callable that renames it.

        *blocks_quit* marks work that quitting must not appear to cut short (a
        launch): the app asks before it waits on it.
        """
        key = next(self._ids)

        def _describe(text: str) -> None:
            with self._lock:
                self._entries[key] = (text, blocks_quit)

        _describe(description)
        try:
            yield _describe
        finally:
            with self._lock:
                self._entries.pop(key, None)

    def descriptions(self, *, blocking_only: bool = False) -> list[str]:
        """What is still running, oldest first."""
        with self._lock:
            return [text for text, blocks in self._entries.values() if blocks or not blocking_only]


@contextmanager
def tracked_work(app: App, description: str, *, blocks_quit: bool = False) -> Iterator[Callable[[str], None]]:
    """``app.background_work.track`` when the app keeps one; a no-op otherwise."""
    tracker = getattr(app, "background_work", None)
    if not isinstance(tracker, BackgroundWork):
        yield lambda _text: None
        return
    with tracker.track(description, blocks_quit=blocks_quit) as describe:
        yield describe


def call_ui(app: App, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """``app.call_from_thread`` that drops the call once the app has stopped; True when it ran.

    A worker can outlive the app (see :class:`BackgroundWork`). After exit
    ``call_from_thread`` raises ``RuntimeError``, and during shutdown the
    callback meets widgets that are already pruned; either way the worker
    raised and Textual printed a traceback over the restored shell. Failures
    are only swallowed when the app is no longer running, so a broken callback
    in a live app still surfaces.
    """
    if not app.is_running:
        return False
    try:
        app.call_from_thread(callback, *args, **kwargs)
    except Exception:
        if app.is_running:
            raise
        return False
    return True


def local_post_commands_refusal(recipe: "Recipe", name: str) -> str | None:
    """Why the monitor will not launch *recipe*, or None when it can.

    ``post_commands`` run on this machine with the terminal handed to them
    (``orchestration/hooks.py``: a ``sudo`` prompt must keep working), which
    would take the terminal from the TUI. The CLI is the place for them.
    """
    if not getattr(recipe, "post_commands", None):
        return None
    command = get_application_profile().command
    return f"This recipe runs post_commands on this machine, which need the terminal; launch it with: {command} run {name}"


@dataclass(frozen=True)
class RecipeLaunch:
    """A validated launch request handed from the params modal to the app."""

    options: RunOptions
    label: str
    """RecipeSummary.name, for notifications."""
    plan: Any = field(default=None, compare=False)
    """The ``RunPlan`` the Run preflight built from exactly ``options``.

    Handed to ``api.run(options, plan=plan)`` so the launch is the placement
    the preflight checked, decided once, without another hardware probe and
    occupancy sweep. ``None`` makes ``run`` plan for itself.
    """


class RecipeActivated(Message):
    """A click on the already-highlighted recipe: open its run configuration."""

    def __init__(self, option_index: int) -> None:
        super().__init__()
        self.option_index = option_index


class _RecipeList(OptionList):
    """Recipe list with click-to-highlight and click-again-to-activate.

    ``OptionList`` selects on every click, which opened the run configuration
    before the info box could follow the pointer. A click here only moves the
    highlight (the screen's ``OptionHighlighted`` handler refreshes the info
    box); a click on the highlighted row posts :class:`RecipeActivated`, and the
    keyboard keeps the base ``enter`` activation through ``OptionSelected``.
    """

    async def _on_click(self, event: events.Click) -> None:
        # Textual dispatches handlers along the whole MRO, so OptionList's own
        # _on_click would still run and select the option outright; this is the
        # documented way to stop base-class handlers.
        event.prevent_default()
        clicked = event.style.meta.get("option")
        if clicked is None or self._options[clicked].disabled:
            return
        if clicked == self.highlighted:
            self.post_message(RecipeActivated(clicked))
            return
        self.highlighted = clicked


def _default_text(value: Any) -> str:
    """Placeholder text for a recipe default (``unset`` when absent)."""
    if value is None or value == "":
        return "unset"
    return str(value)


def _runtime_family(runtime: str, sctx: Any = None) -> str:
    name = runtime.lower()
    try:
        return get_runtime(name, v=sctx.variables if sctx is not None else None).get_family().lower() or name
    except Exception:  # noqa: BLE001 - recipes for unavailable runtimes must remain browsable
        return RUNTIME_DISPLAY.get(name, name)


@dataclass(frozen=True)
class RuntimeChecks:
    """Pre-flight result the checks interstitial hands to the parameter modal."""

    scheduler: str
    """Resolved scheduler name (``RunPlan.scheduler``)."""
    selected: tuple[str, ...]
    """Hosts the scheduler placed the workload on (``RunPlan.host_list``)."""
    selectable: tuple[str, ...]
    """Candidates not rejected by the scheduler, reachability, or memory checks."""
    notes: tuple[str, ...] = ()
    """Placement notes from the plan, rendered verbatim."""
    warnings: tuple[str, ...] = ()
    """Advisory check outcomes (probe unavailable, aggregate fit warnings)."""
    reasons: tuple[tuple[str, str], ...] = ()
    """Non-selectable hosts and why, as ``(host, short reason)`` pairs."""
    ok: bool = True
    """False when the plan itself failed (no placement exists to launch)."""
    plan: Any = field(default=None, compare=False)
    """The ``RunPlan`` behind ``selected``, when planning succeeded."""


# Scheduler rejections are identifiers (``_occupancy_base._rejections``); the
# checkbox labels need something a person reads.
_REJECTION_LABELS = {
    "occupied": "no free slot",
    "exclusive_owner": "in use",
    "compute_budget": "no compute budget",
    "memory_budget": "no headroom",
    "memory_unknown": "memory unknown",
    "observation_failed": "unavailable",
    "no_accelerators": "no GPU",
    "unknown_gpu_indices": "GPU occupancy unknown",
    "same_request": "layout repeats a GPU",
}


class RecipePickerScreen(Screen[RecipeLaunch | None]):
    """Searchable recipe catalog: filter and metadata row, search bar, list."""

    DEFAULT_CSS = """
    RecipePickerScreen {
        layout: vertical;
    }
    # A one-row Input keeps only its border: the text and cursor are clipped
    # away. Three rows is the widget's own default (tall border + text line).
    #recipe-search {
        height: 3;
        margin: 0 1;
    }
    #recipe-all-registries {
        margin: 0 1;
    }
    # The filter and metadata boxes share one row, so they share one height:
    # both stretch to the row, and the row scales with the terminal.
    #recipe-cols {
        height: 34%;
        margin: 1 1 0 1;
    }
    #recipe-filters {
        width: 20%;
        height: 100%;
        border: round $accent;
    }
    #recipe-meta {
        width: 80%;
        height: 100%;
        border: round $accent;
        padding: 0 1;
        /* Identity and stats can still overflow a very short terminal; the box
           scrolls so nothing becomes unreachable. */
        overflow-y: auto;
    }
    #recipe-meta-head {
        height: auto;
    }
    /* The description owns everything the identity line and the stats do not:
       its height is 1fr between two auto-height children. */
    #recipe-meta-desc {
        height: 1fr;
        min-height: 1;
        overflow-y: auto;
        /* The description is supporting prose; the decision fields stay lit. */
        text-style: dim;
    }
    /* Everything a decision needs above the prose; the description is last. */
    #recipe-meta-stats {
        height: auto;
    }
    /* Search and list sit under the boxes and span the full width. */
    #recipe-list {
        height: 1fr;
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        # priority: the app-level escape/q bindings quit the app; the screen must
        # swallow both (same reason DumpLogScreen binds escape).
        Binding("escape", "cancel", "Close", priority=True),
        Binding("q", "cancel", "Close", show=False, priority=True),
    ]

    def __init__(
        self,
        cluster: str | None,
        hosts: tuple[str, ...],
        sctx: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cluster: str | None = cluster
        self.hosts: tuple[str, ...] = tuple(hosts)
        self.sctx: Any = sctx
        self._recipes: list[RecipeSummary] = []
        self._runtime_families: dict[str, str] = {}
        self._shown: list[RecipeSummary] = []
        self._filter: str = "all"
        self._loaded: bool = False
        self._catalog_request = 0

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        # Top row: filters and metadata side by side, equal height. Search and
        # list each own a full-width row below them.
        with Horizontal(id="recipe-cols"):
            yield OptionList(id="recipe-filters")
            with Vertical(id="recipe-meta"):
                yield Static(id="recipe-meta-head", markup=True)
                yield Static(id="recipe-meta-stats", markup=True)
                # Last field by requirement: the prose closes the box.
                yield Static(id="recipe-meta-desc", markup=True)
        yield Checkbox(
            "All registries",
            id="recipe-all-registries",
            tooltip="Include enabled registries hidden from default searches (Stemless: no).",
        )
        yield Input(id="recipe-search", placeholder="Search recipes (name, model, description)…")
        yield _RecipeList(id="recipe-list")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#recipe-search", Input).focus()
        self._reload_catalog()

    # -- catalog loading ----------------------------------------------------

    def _reload_catalog(self) -> None:
        self._catalog_request += 1
        self._loaded = False
        self._shown = []
        self.query_one("#recipe-filters", OptionList).disabled = True
        self.query_one("#recipe-meta-head", Static).update("[dim]Select a recipe to see details[/dim]")
        self.query_one("#recipe-meta-stats", Static).update("")
        self.query_one("#recipe-meta-desc", Static).update("")
        listed = self.query_one("#recipe-list", OptionList)
        listed.clear_options()
        listed.add_option(Option("Loading recipes…", id="loading", disabled=True))
        self._load_worker(self._catalog_request, self.query_one("#recipe-all-registries", Checkbox).value)

    @work(thread=True, exclusive=True, group="recipes")
    def _load_worker(self, request: int, include_hidden: bool) -> None:
        """Fetch the catalog off the UI thread; first use may clone registries."""
        app = self.app
        failure = None
        with tracked_work(app, "the recipe catalog load"):
            try:
                recipes = search_recipes(include_hidden=include_hidden, sctx=self.sctx)
            except Exception as error:  # noqa: BLE001 - a catalog failure must not break the app
                failure = str(error)
                recipes = []
            families = {name: _runtime_family(name, self.sctx) for name in {s.runtime.lower() for s in recipes if s.runtime}}
        call_ui(app, self._on_loaded, recipes, request, families, failure)

    def _on_loaded(self, recipes: list[RecipeSummary], request: int, families: dict[str, str], failure: str | None = None) -> None:
        # The screen can be dismissed while the catalog is still loading (the
        # first search may clone registries for minutes); the worker's callback
        # is already queued and still runs after the pop, where querying the
        # widgets would raise NoMatches on the UI thread. A popped Screen keeps
        # is_mounted True, so is_attached is the state that goes false.
        # Cancelling a threaded worker does not stop its catalog call. A late
        # result from an earlier toggle must not replace the requested scope.
        if not self.is_attached or request != self._catalog_request:
            return
        self._recipes = list(recipes)
        self._runtime_families = families
        self._loaded = True
        self._rebuild_filters()
        self.query_one("#recipe-filters", OptionList).disabled = False
        self._rebuild_list()
        if failure is not None:
            self.app.notify(f"Recipe search failed: {failure}", severity="error")

    def _rebuild_filters(self) -> None:
        filters = self.query_one("#recipe-filters", OptionList)
        options = [Option("All recipes", id="all")]
        for runtime in sorted(set(self._runtime_families.values())):
            options.append(Option(Content(f"Runtime: {runtime}"), id=f"runtime:{runtime}"))
        if any(s.registry is None for s in self._recipes):
            options.append(Option("Local recipes", id="registry:"))
        for registry in sorted({s.registry for s in self._recipes if s.registry}):
            options.append(Option(Content(f"@{registry}"), id=f"registry:{registry}"))
        selected = next((i for i, option in enumerate(options) if option.id == self._filter), 0)
        self._filter = options[selected].id or "all"
        with self.prevent(OptionList.OptionHighlighted):
            filters.clear_options()
            filters.add_options(options)
            filters.highlighted = selected

    # -- list rebuilding ----------------------------------------------------

    def _facet_matches(self, summary: RecipeSummary) -> bool:
        facet = self._filter
        if facet == "all":
            return True
        kind, _, value = facet.partition(":")
        if kind == "runtime":
            return self._runtime_families.get(summary.runtime.lower(), summary.runtime.lower()) == value
        if kind == "registry":
            return (summary.registry or "") == value
        return True

    def _current_name(self) -> str | None:
        listed = self.query_one("#recipe-list", OptionList)
        if listed.highlighted is not None and 0 <= listed.highlighted < len(self._shown):
            return self._shown[listed.highlighted].name
        return None

    def _rebuild_list(self) -> None:
        if not self._loaded:
            return
        keep = self._current_name()
        query = self.query_one("#recipe-search", Input).value.strip()
        self._shown = [s for s in self._recipes if self._facet_matches(s) and recipe_matches_query(s.to_dict(), query)]

        listed = self.query_one("#recipe-list", OptionList)
        listed.clear_options()
        if not self._shown:
            listed.add_option(Option("[dim]No recipes match[/dim]", id="empty", disabled=True))
            return
        # Index ids, not names: two registry recipes can share one qualified name.
        for idx, summary in enumerate(self._shown):
            prompt = f"{summary.name} — {summary.model}" if summary.model else summary.name
            listed.add_option(Option(Content(prompt), id=str(idx)))

        target = next((i for i, s in enumerate(self._shown) if s.name == keep), 0)
        listed.highlighted = target
        self._show_meta(str(target))

    def _show_meta(self, option_id: str | None) -> None:
        try:
            idx = int(option_id or "")
        except ValueError:
            return
        if 0 <= idx < len(self._shown):
            head, stats, desc = self._render_meta(self._shown[idx])
            self.query_one("#recipe-meta-head", Static).update(head)
            self.query_one("#recipe-meta-stats", Static).update(stats)
            self.query_one("#recipe-meta-desc", Static).update(desc)

    @staticmethod
    def _render_meta(summary: RecipeSummary) -> tuple[Content, Content, Content]:
        """Info box fields, in render order: identity, decision stats, prose.

        Only what a launch decision needs: which recipe (the name carries its
        registry qualifier), what it runs, how many hosts it needs, and the
        parallelism and memory knobs. Paths and other provenance internals are
        left out. The description closes the box.

        Recipe fields are registry content, so they are substituted as plain
        text: parsed as markup, a ``[/think]`` in a description raised at
        render time and exited the app.
        """
        tp = summary.tensor_parallel if summary.tensor_parallel is not None else "runtime default"
        gpu_mem = summary.gpu_memory_utilization if summary.gpu_memory_utilization is not None else "recipe default"
        head = Content.from_markup(
            "[bold]$name[/bold]\nmodel: $model\nruntime: $runtime",
            name=summary.name,
            model=summary.model or "-",
            runtime=summary.runtime or "-",
        )
        stats = Content(f"min nodes: {summary.min_nodes} · tensor parallel: {tp} · gpu memory util: {gpu_mem}")
        desc = Content(summary.description) if summary.description else Content.from_markup("[dim]no description[/dim]")
        return head, stats, desc

    # -- events -------------------------------------------------------------

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "recipe-all-registries":
            self._reload_catalog()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "recipe-search":
            self._rebuild_list()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "recipe-search":
            self.query_one("#recipe-list", OptionList).focus()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id == "recipe-filters":
            self._filter = event.option.id or "all"
            self._rebuild_list()
        elif event.option_list.id == "recipe-list":
            self._show_meta(event.option.id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "recipe-filters":
            self.query_one("#recipe-list", OptionList).focus()
        elif event.option_list.id == "recipe-list":
            self._open_params(event.option.id)

    def on_recipe_activated(self, event: RecipeActivated) -> None:
        # Second click on the row that is already highlighted (keyboard enter
        # still arrives as OptionSelected above).
        self._open_params(str(event.option_index))

    def _open_params(self, option_id: str | None) -> None:
        try:
            idx = int(option_id or "")
        except ValueError:
            return
        if not (0 <= idx < len(self._shown)):
            return
        summary = self._shown[idx]
        try:
            # Resolve by path, not name: the summary's path is the exact file, so
            # same-stem recipes across registries cannot turn into an ambiguity error.
            recipe = resolve_recipe(summary.path, sctx=self.sctx)
        except Exception as error:  # noqa: BLE001 - keep the picker usable
            self.app.notify(f"Cannot load recipe: {error}", severity="error")
            return
        try:
            # Hooks from an untrusted source make api.run refuse to launch without
            # RunOptions(trust=True); ask here, where the user can authorize.
            pending_hooks = untrusted_hooks(recipe, sctx=self.sctx)
        except Exception:  # noqa: BLE001 - trust discovery must not block the picker
            pending_hooks = {}
        self.app.push_screen(
            RuntimeChecksScreen(
                recipe=recipe,
                summary=summary,
                cluster=self.cluster,
                hosts=self.hosts,
                sctx=self.sctx,
            ),
            callback=partial(self._checks_done, recipe, summary, pending_hooks),
        )

    def _checks_done(
        self,
        recipe: "Recipe",
        summary: RecipeSummary,
        pending_hooks: dict[str, tuple[str, ...]],
        checks: RuntimeChecks | None,
    ) -> None:
        # The picker can be gone by the time the interstitial closes (the app
        # may have been dismissed while checks ran); querying widgets then
        # raises NoMatches on the UI thread.
        if not self.is_attached:
            return
        if checks is None:
            # Cancel: no launch is possible, so hand focus back to the list
            # without ever opening the params modal.
            self.query_one("#recipe-list", OptionList).focus()
            return
        self.app.push_screen(
            RunParamsScreen(
                recipe=recipe,
                summary=summary,
                cluster=self.cluster,
                hosts=self.hosts,
                checks=checks,
                pending_hooks=pending_hooks,
                sctx=self.sctx,
            ),
            callback=self._params_done,
        )

    def _params_done(self, launch: RecipeLaunch | None) -> None:
        if launch is None:
            # Cancel/escape: the modal closed, hand focus back to the recipe list.
            self.query_one("#recipe-list", OptionList).focus()
            return
        self.dismiss(launch)

    def action_cancel(self) -> None:
        self.dismiss(None)


class RuntimeChecksScreen(ModalScreen[RuntimeChecks | None]):
    """Progress-only pre-flight modal: a bar plus Cancel while checks run.

    ``api.plan`` performs cluster-facing work (hardware discovery, one
    occupancy query, scheduling), so it runs in a worker under an indeterminate
    progress bar. Completing the checks dismisses with the :class:`RuntimeChecks`
    and the picker immediately opens the parameter modal; Cancel dismisses with
    ``None`` and the worker's result is discarded. The parameter modal reuses it
    as a preflight: it passes the final ``RunOptions`` so the plan sees the
    knobs, and a failure there returns to the settings instead of launching;
    success hands its plan to the launch. A failed plan in the picker flow is
    advisory: the Run preflight re-plans from the parameters the next modal
    collects and feasibility can change with them (a lower tensor parallel may
    fit where the recipe default did not), so the parameters modal opens either
    way.
    """

    DEFAULT_CSS = """
    RuntimeChecksScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }
    #checks-dialog {
        width: 60;
        max-width: 100%;
        height: auto;
        border: thick $accent;
        background: $background-lighten-2;
        padding: 1 2;
    }
    #checks-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #checks-progress {
        width: 100%;
        margin-bottom: 1;
    }
    #checks-buttons {
        height: auto;
    }
    """

    BINDINGS = [
        # priority: the app-level escape/q bindings quit the app while a screen
        # is up; escaping the interstitial must only cancel it.
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("q", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(
        self,
        recipe: "Recipe",
        summary: RecipeSummary,
        cluster: str | None,
        hosts: tuple[str, ...],
        sctx: Any = None,
        options: RunOptions | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.recipe: Recipe = recipe
        self.summary = summary
        self.cluster = cluster
        self.hosts: tuple[str, ...] = tuple(hosts)
        self.sctx = sctx
        # A preflight from the parameter modal carries the final options: the
        # plan then sees the knobs Run was pressed with, not the recipe defaults.
        self.options = options

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="checks-dialog"):
            yield Static("Querying scheduler...", id="checks-title")
            # total=None: the long pole is one SSH-bound plan call, so a real
            # percentage would sit at zero; an indeterminate bar shows activity.
            yield ProgressBar(show_eta=False, show_percentage=False, id="checks-progress")
            with Horizontal(id="checks-buttons"):
                yield Button("Cancel", id="checks-cancel")

    def on_mount(self) -> None:
        self.query_one("#checks-cancel", Button).focus()
        self._checks_worker()

    # -- checks -------------------------------------------------------------

    @work(thread=True, exclusive=True, group="runtime-checks")
    def _checks_worker(self) -> None:
        """Plan the launch, then check each candidate host's availability and memory fit off the UI thread."""
        # Cancel only stops Textual waiting on this thread: the plan keeps
        # running, and its result lands on a detached screen that ignores it.
        app = self.app
        with tracked_work(app, f"the scheduler check for {self.summary.name}"):
            outcome = self._run_checks()
        call_ui(app, *outcome)

    def _run_checks(self) -> tuple[Any, ...]:
        """``(handler, *args)`` for the UI thread: the checks, or why planning failed."""
        try:
            options = self.options or RunOptions(recipe=deepcopy(self.recipe), cluster=self.cluster, hosts=self.hosts or None)
            run_plan = plan(options, sctx=self.sctx)
        except (InsufficientCapacity, HostsUnreachable) as error:
            # The plan failed on capacity, not on an unknown: the scheduler's
            # own per-host rejections say which rows are dead ends.
            return self._checks_failed, str(error), self._blocked_checks(error)
        except Exception as error:  # noqa: BLE001 - the modal owns the failure, the app must survive it
            return self._checks_failed, str(error)

        selected = tuple(run_plan.host_list)
        scheduler = run_plan.scheduler
        notes = tuple(run_plan.notes)
        warnings: list[str] = []
        reasons: dict[str, str] = {}
        cluster = getattr(run_plan, "cluster", None)

        # The scheduler owns capacity admission, including fractional GPU
        # allocations and reuse of the same intent. Zero *unused* slots does
        # not invalidate its placement. This probe only checks reachability.
        if self.cluster:
            try:
                capacity = catalog_cluster_capacity(self.cluster, sctx=self.sctx)
            except Exception as error:  # noqa: BLE001 - advisory probe; unknown availability must not block
                warnings.append(f"Could not check host availability ({error}); availability is not filtered")
            else:
                by_host = {entry["host"]: entry for entry in capacity.get("hosts", [])}
                for host in self.hosts:
                    entry = by_host.get(host)
                    if entry is None or not entry["reachable"]:
                        if entry is not None:
                            reasons[host] = "unavailable"
                        continue

        if cluster is not None:
            try:
                # A preflight plans the resolved options, so the estimate must
                # use that recipe and its overrides, not the picker's defaults.
                # (Options may carry a recipe *name*; only a loaded Recipe can
                # answer estimate_vram, and the preflight always passes one.)
                recipe = options.recipe if not isinstance(options.recipe, str) else deepcopy(self.recipe)
                overrides = options.overrides
                estimate = recipe.estimate_vram(
                    cli_overrides=overrides,
                    auto_detect=not is_local_model_path(recipe.model),
                )
                fit = check_fit(estimate, cluster)
            except Exception as error:  # noqa: BLE001 - fit is advisory: an unknown capacity must not block a launch
                warnings.append(f"Could not check memory fit ({error}); every host stays selectable")
            else:
                # No detail means the host was not part of the fit report, which
                # is not evidence of a problem; an explicit !ok is.
                for host, detail in fit.per_host.items():
                    if not detail.ok:
                        reasons.setdefault(host, "no headroom")
                warnings.extend(fit.warnings)
        selectable = tuple(host for host in self.hosts if host not in reasons)

        checks = RuntimeChecks(
            scheduler=scheduler,
            selected=selected,
            selectable=selectable,
            notes=notes,
            warnings=tuple(warnings),
            reasons=tuple(reasons.items()),
            plan=run_plan,
        )
        return self._checks_ready, checks

    def _checks_ready(self, checks: RuntimeChecks) -> None:
        # Cancel may have closed the modal while the worker ran; the picker's
        # callback must not open the params modal for a discarded check.
        if not self.is_attached:
            return
        self.dismiss(checks)

    def _blocked_checks(self, error: Exception) -> RuntimeChecks:
        """Turn a capacity or reachability failure into disabled host rows.

        A plan that fails because the scheduler rejected every candidate is a
        dead end, not an unknown: the checkboxes must reflect it instead of
        offering a launch that will fail. ``HostsUnreachable`` disables only the
        hosts it names; ``InsufficientCapacity`` disables the hosts in its
        ``rejections`` and falls back to every candidate when the error carries
        no per-host attribution (the request itself is infeasible).
        """
        if isinstance(error, HostsUnreachable):
            blocked = {host: "unavailable" for host in getattr(error, "hosts", ())}
        else:
            blocked = {}
            for rejection in getattr(error, "rejections", ()) or ():
                host = getattr(rejection, "host", None)
                reason = getattr(rejection, "reason", "") or ""
                if host in self.hosts and host not in blocked:
                    blocked[host] = _REJECTION_LABELS.get(reason, reason.replace("_", " ") or "insufficient resources")
            if not blocked:
                blocked = {host: "insufficient capacity" for host in self.hosts}
        return RuntimeChecks(
            scheduler="",
            selected=(),
            selectable=tuple(host for host in self.hosts if host not in blocked),
            warnings=(str(error),),
            reasons=tuple((host, blocked[host]) for host in self.hosts if host in blocked),
            ok=False,
        )

    def _checks_failed(self, message: str, checks: RuntimeChecks | None = None) -> None:
        # In the picker flow this is advisory: the Run preflight re-plans from the
        # final RunOptions, so a plan that fails on the recipe defaults (say, TP too
        # high for the free capacity) is recoverable by editing the knobs in the
        # next modal. Without a plan there is no fit data, so every host stays
        # selectable and pre-checked, exactly like the CLI's default candidate
        # list — but a capacity failure arrives with per-host rejections and
        # uses those. ``ok=False`` tells the parameter modal's own preflight
        # that there is no placement to launch.
        if not self.is_attached:
            return
        self.dismiss(
            checks
            or RuntimeChecks(
                scheduler="",
                selected=self.hosts,
                selectable=self.hosts,
                warnings=(message,),
                ok=False,
            )
        )

    # -- actions ------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "checks-cancel":
            self.action_cancel()

    def action_cancel(self) -> None:
        # The worker may be inside an SSH-bound plan call; cancelling the group
        # drops the result, and the is_attached guard in the handlers keeps a
        # late completion from opening the params modal.
        self.workers.cancel_group(self, "runtime-checks")
        self.dismiss(None)


@dataclass(frozen=True)
class _Knob:
    """One advanced control in the run-parameters modal."""

    key: str
    """RunOptions field or serve-override key."""
    label: str
    kind: str = "text"
    """text | int | float | flag | offline."""
    column: str = "long"
    """short (20% column) | long (80% column)."""
    placeholder: str = ""
    default: bool = False
    """Initial value of a flag."""


# Every launch knob ``api.RunOptions`` accepts from the command line, minus the
# ones this modal already owns (recipe, hosts, cluster, overrides, trust) and
# the ones the cluster file decides (transfer mode/interface, topology, ports).
# Serve-side keys without a typed field stay reachable through each recipe
# default's own field and the free ``-o key=value`` input.
_LAUNCH_KNOBS: tuple[_Knob, ...] = (
    _Knob("pipeline_parallel", "Pipeline parallel", "int", "short"),
    _Knob("data_parallel", "Data parallel", "int", "short"),
    _Knob("scheduler", "Scheduler", "text", "short", "default"),
    _Knob("workload_id_override", "Workload id", "text", "short", "auto"),
    _Knob("image", "Image", "text", "long", "recipe"),
    _Knob("served_model_name", "Served model", "text", "long", "recipe"),
    _Knob("cache_dir", "Cache dir", "text", "long", "default"),
    _Knob("memory_limit", "Memory limit", "text", "long", "e.g. 32G"),
    _Knob("labels", "Labels", "text", "long", "key=value …"),
    _Knob("extra_docker_opts", "Docker opts", "text", "long", "extra args"),
    _Knob("offline", "Offline", "offline", "long"),
    _Knob("dry_run", "Dry run", "flag", "long"),
    _Knob("ensure", "Reuse job", "flag", "long"),
    _Knob("sync_tuning", "Sync tuning", "flag", "long", default=True),
    _Knob("rootful", "Rootful", "flag", "long"),
    _Knob("auto_port", "Auto port", "flag", "long"),
)

# Recipe defaults that the always-visible fields already own; a second field
# for the same key would be a second source of truth.
_COMMON_DEFAULT_KEYS = frozenset({"tensor_parallel", "gpu_memory_utilization", "max_model_len", "port", "served_model_name"})


class RunParamsScreen(ModalScreen[RecipeLaunch | None]):
    """Common runtime parameters, every recipe/launch knob, and raw overrides."""

    DEFAULT_CSS = """
    RunParamsScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }
    #params-dialog {
        width: 96;
        max-width: 100%;
        height: auto;
        max-height: 90%;
        border: thick $accent;
        background: $background-lighten-2;
        padding: 1 2;
        overflow-y: auto;
    }
    #param-title {
        text-style: bold;
        margin-bottom: 1;
    }
    .param-label {
        margin-top: 1;
    }
    #param-cols {
        height: auto;
    }
    #param-short {
        width: 20%;
        height: auto;
        padding-right: 2;
    }
    /* The 20% column is narrow; one cell of horizontal padding buys back a
       character of placeholder text without touching the other inputs. */
    #param-short Input {
        padding: 0 1;
    }
    #param-long {
        width: 80%;
        height: auto;
        padding-left: 1;
    }
    #param-adv-short, #param-adv-long {
        display: none;
        height: auto;
    }
    #param-adv-short.-shown, #param-adv-long.-shown {
        display: block;
    }
    #param-error {
        height: auto;
        margin-top: 1;
    }
    #param-hosts Checkbox {
        width: 1fr;
    }
    #param-hosts-note {
        height: auto;
    }
    # Error and buttons share one docked footer: a short terminal otherwise
    # scrolls the validation message underneath the button row, where the dock
    # hides it and the Run press looks like it did nothing.
    #param-footer {
        dock: bottom;
        height: auto;
    }
    #param-buttons {
        height: auto;
    }
    #param-buttons Button {
        width: auto;
        min-width: 0;
        padding: 0 1;
        margin-right: 1;
    }
    # The Advanced toggle shares the action row with the buttons; like the
    # buttons it gets a gap so the row does not read as one control.
    #param-buttons Checkbox {
        margin-right: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    def __init__(
        self,
        recipe: "Recipe",
        summary: RecipeSummary,
        cluster: str | None,
        hosts: tuple[str, ...],
        checks: RuntimeChecks,
        pending_hooks: dict[str, tuple[str, ...]] | None = None,
        sctx: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.recipe: Recipe = recipe
        self.summary = summary
        self.cluster = cluster
        self.hosts: tuple[str, ...] = tuple(hosts)
        self.checks = checks
        self.sctx = sctx
        # Hooks api.run would refuse without RunOptions(trust=True); empty for
        # local recipes and trusted registries.
        self.pending_hooks: dict[str, tuple[str, ...]] = dict(pending_hooks or {})
        # Advanced controls start hidden and their values are ignored until the
        # Advanced checkbox reveals them.
        self._advanced = False
        self._default_knobs: list[tuple[str, Any]] = []
        self._env_keys: list[str] = []
        # The options a Run press built; the preflight's callback launches them
        # only if the scheduler still accepts every candidate host.
        self._pending_options: RunOptions | None = None
        self._refreshing_hosts = False
        # The host boxes as this modal last set them from a scheduler check, and
        # whether the user has toggled one since. Untouched boxes only *show*
        # the placement; launching with them as the candidate list would make
        # the display authoritative over placement (see _candidate_hosts).
        self._default_selection = self._eligible_selection(checks, checks.selected)
        self._hosts_edited = False

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        defaults = getattr(self.recipe, "defaults", None) or {}
        # Only scalar recipe defaults become fields: -o key=value is exactly
        # their override form, and a mapping default is still reachable through
        # the free overrides input in dotted (key.field=value) form. Keys a
        # dedicated control already owns must not get a second source of truth.
        owned = _COMMON_DEFAULT_KEYS | {knob.key for knob in _LAUNCH_KNOBS}
        self._default_knobs = [
            (str(key), value) for key, value in defaults.items() if key not in owned and isinstance(value, (str, int, float, bool))
        ]
        env = getattr(self.recipe, "env", None) or {}
        self._env_keys = sorted(env)
        with Vertical(id="params-dialog"):
            yield Static(Content.from_markup("Run [bold]$name[/bold]?", name=self.summary.name), id="param-title")
            with Horizontal(id="param-cols"):
                with Vertical(id="param-short"):
                    yield Static("Tensor parallel", classes="param-label")
                    yield Input(placeholder=_default_text(defaults.get("tensor_parallel")), id="param-tp", type="integer", valid_empty=True)
                    yield Static("GPU memory util", classes="param-label")
                    yield Input(
                        placeholder=_default_text(defaults.get("gpu_memory_utilization")),
                        id="param-gpu-mem",
                        type="number",
                        valid_empty=True,
                    )
                    yield Static("Max model len", classes="param-label")
                    yield Input(
                        placeholder=_default_text(defaults.get("max_model_len")), id="param-max-len", type="integer", valid_empty=True
                    )
                    yield Static("Serve port", classes="param-label")
                    yield Input(placeholder=_default_text(defaults.get("port")), id="param-port", type="integer", valid_empty=True)
                    with Vertical(id="param-adv-short"):
                        yield from self._knob_fields(knob for knob in _LAUNCH_KNOBS if knob.column == "short")
                with Vertical(id="param-long"):
                    if self.hosts:
                        scheduler = self.checks.scheduler
                        title = f"Hosts ({scheduler})" if scheduler else "Hosts (unavailable)"
                        yield Static(title, id="param-hosts-title", classes="param-label")
                        checked_count = len([host for host in self.checks.selected if host in self.checks.selectable])
                        note = " · ".join(
                            [
                                f"{checked_count}/{len(self.hosts)} selected · {self.checks.scheduler or 'default'}",
                                *self.checks.notes,
                                *self.checks.warnings,
                            ]
                        )
                        yield Static(note, id="param-hosts-note", markup=False)
                        with ItemGrid(id="param-hosts", min_column_width=30):
                            reasons = dict(self.checks.reasons)
                            for index, host in enumerate(self.hosts):
                                eligible = host in self.checks.selectable
                                label = Content(host if eligible else f"{host} ({reasons.get(host, 'insufficient resources')})")
                                yield Checkbox(
                                    label,
                                    # A disabled row cannot be unchecked, so an
                                    # over-capacity host the plan pre-selected
                                    # must not start checked; it is left off and
                                    # the scheduler re-plans from the checked set.
                                    value=eligible and host in self.checks.selected,
                                    disabled=not eligible,
                                    id=f"param-host-{index}",
                                )
                    if self.pending_hooks:
                        yield Static("Untrusted recipe needs authorization", classes="param-label")
                        # The summary is recipe content, so it is never parsed as markup,
                        # and the checkbox names nothing the recipe can influence.
                        yield Static(self._trust_summary(), id="param-trust-summary", markup=False)
                        yield Checkbox("Authorize the hooks and overrides above", id="param-trust")
                    with Vertical(id="param-adv-long"):
                        yield from self._knob_fields(knob for knob in _LAUNCH_KNOBS if knob.column == "long")
                        for index, (key, value) in enumerate(self._default_knobs):
                            yield Static(f"{key} (default)", classes="param-label")
                            yield Input(placeholder=str(value), id=f"param-default-{index}")
                        for index, key in enumerate(self._env_keys):
                            yield Static(f"env.{key}", classes="param-label")
                            yield Input(placeholder=str(env[key]), id=f"param-env-{index}")
                        yield Static("Overrides (-o key=value)", classes="param-label")
                        yield Input(placeholder="key=value …", id="param-advanced")
            with Vertical(id="param-footer"):
                yield Static(id="param-error")
                with Horizontal(id="param-buttons"):
                    yield Checkbox("Advanced", id="param-advanced-toggle")
                    yield Button("Check hosts", id="param-recheck")
                    yield Button("Cancel", id="param-cancel")
                    yield Button("Run", id="param-run", variant="primary")

    def on_mount(self) -> None:
        # Say so before any parameters are typed: Run will refuse this recipe.
        refusal = local_post_commands_refusal(self.recipe, self.summary.name)
        if refusal is not None:
            self._set_error(refusal)

    @classmethod
    def _knob_fields(cls, knobs: Iterable[_Knob]) -> ComposeResult:
        """Widgets for *knobs*: a checkbox per flag, a Select for tri-state, typed Inputs otherwise."""
        for knob in knobs:
            widget_id = f"param-knob-{knob.key}"
            if knob.kind == "flag":
                yield Checkbox(knob.label, value=knob.default, id=widget_id)
                continue
            yield Static(knob.label, classes="param-label")
            if knob.kind == "offline":
                yield Select(
                    (("default", "inherit"), ("offline", "offline"), ("online", "online")),
                    value="inherit",
                    allow_blank=False,
                    id=widget_id,
                )
            else:
                input_kwargs: dict[str, Any] = {"valid_empty": True}
                if knob.kind == "int":
                    input_kwargs["type"] = "integer"
                elif knob.kind == "float":
                    input_kwargs["type"] = "number"
                yield Input(placeholder=knob.placeholder, id=widget_id, **input_kwargs)

    def _trust_summary(self) -> str:
        """What the trust checkbox authorizes: the same summary ``sparkrun run`` prompts with.

        ``RunOptions(trust=True)`` authorizes the recipe's container-isolation
        overrides (host mounts, privileged mode, executor selection, pre-placed
        paths) as well as its hooks, so both are listed. Listing only the hooks
        let one tick grant mounts the user never saw.
        """
        from sparkrun.core.launcher import recipe_isolation_overrides
        from sparkrun.utils.text import format_hook_summary

        try:
            isolation = recipe_isolation_overrides(self.recipe)
        except Exception:  # noqa: BLE001 - disclosure must not break the modal
            isolation = ["Could not list this recipe's container-isolation overrides; trusting it allows any it declares."]
        return format_hook_summary(self.summary.name, self.pending_hooks, isolation=isolation)

    # -- hosts --------------------------------------------------------------

    def _eligible_selection(self, checks: RuntimeChecks, selected: Iterable[str]) -> frozenset[str]:
        """The hosts a box shows checked: chosen and still eligible under *checks*."""
        chosen = set(selected)
        return frozenset(host for host in self.hosts if host in checks.selectable and host in chosen)

    def _candidate_hosts(self, selected: tuple[str, ...]) -> tuple[str, ...]:
        """The candidate list the launch hands the scheduler.

        Pre-checked boxes are the scheduler's own placement. Passing them back
        as ``RunOptions.hosts`` narrows the candidate set to that placement, and
        ``api.run`` derives a greedy cluster_id from, and evicts the superseded
        deployment within, the candidate set only: a relaunch could then miss
        the copy it replaces on an unchecked host, or get a cluster_id that
        ``sparkrun run`` / ``stop`` / ``status`` never derive. Untouched boxes
        therefore launch with every eligible monitored host; with none
        disabled that is exactly what ``sparkrun run`` gets from the monitor's
        cluster/hosts flags. Disabled rows (unreachable, no memory headroom)
        stay out, as their boxes say: the scheduler does not check memory fit,
        so offering them would only get the launch placed where this dialog
        then refuses it. A selection the user actually changed is a candidate
        list of their own, like ``--hosts a,b``.
        """
        if self._hosts_edited and frozenset(selected) != self._default_selection:
            return selected
        return tuple(host for host in self.hosts if host in self.checks.selectable)

    def _host_boxes(self) -> list[tuple[str, Checkbox]]:
        return [(host, self.query_one(f"#param-host-{index}", Checkbox)) for index, host in enumerate(self.hosts)]

    def _update_hosts_note(self) -> None:
        checked = sum(1 for _host, box in self._host_boxes() if box.value)
        # Count, scheduler, and placement notes only: warnings are long and the
        # preflight failure already repeats them in #param-error.
        note = " · ".join(
            [
                f"{checked}/{len(self.hosts)} selected · {self.checks.scheduler or 'default'}",
                *self.checks.notes,
            ]
        )
        try:
            self.query_one("#param-hosts-note", Static).update(note)
        except Exception:  # noqa: BLE001 - the note is cosmetic
            pass

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        checkbox_id = event.checkbox.id or ""
        if checkbox_id == "param-advanced-toggle":
            self._advanced = event.value
            for target in ("#param-adv-short", "#param-adv-long"):
                self.query_one(target).set_class(event.value, "-shown")
            return
        if checkbox_id.startswith("param-host-"):
            # Programmatic updates are posted under prevent() (_apply_checks), so
            # every change that reaches here is the user's.
            self._hosts_edited = True
            self._update_hosts_note()

    # -- actions ------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "param-cancel":
            self.dismiss(None)
        elif event.button.id == "param-run":
            self._try_launch()
        elif event.button.id == "param-recheck":
            self._try_launch(refresh_hosts=True)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _set_error(self, message: str) -> None:
        try:
            # The message quotes user input and plan errors; substitute it as text.
            self.query_one("#param-error", Static).update(Content.from_markup("[red]$message[/red]", message=message) if message else "")
        except Exception:
            pass

    def _int_field(self, widget_id: str, label: str) -> int | None:
        text = self.query_one(f"#{widget_id}", Input).value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"{label} must be an integer, got {text!r}") from None

    def _float_field(self, widget_id: str, label: str) -> float | None:
        text = self.query_one(f"#{widget_id}", Input).value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{label} must be a number, got {text!r}") from None

    def _advanced_tokens(self) -> list[str]:
        """Free-form override tokens; empty while Advanced is off."""
        if not self._advanced:
            return []
        raw = self.query_one("#param-advanced", Input).value.strip()
        if not raw:
            return []
        try:
            tokens = [token for token in shlex.split(raw) if token != "-o"]
        except ValueError as error:
            raise ValueError(f"Advanced args are malformed: {error}") from None
        for token in tokens:
            if "=" not in token:
                raise ValueError(f"Advanced args must be key=value, e.g. max_num_seqs=256 (got {token!r})")
        return tokens

    def _knob_values(self) -> dict[str, Any]:
        """Values of the revealed advanced knobs; empty while Advanced is off."""
        if not self._advanced:
            return {}
        values: dict[str, Any] = {}
        for knob in _LAUNCH_KNOBS:
            widget_id = f"#param-knob-{knob.key}"
            if knob.kind == "flag":
                values[knob.key] = self.query_one(widget_id, Checkbox).value
            elif knob.kind == "offline":
                choice = str(self.query_one(widget_id, Select).value)
                values[knob.key] = {"offline": True, "online": False}.get(choice)
            elif knob.kind == "int":
                text = self.query_one(widget_id, Input).value.strip()
                try:
                    values[knob.key] = int(text) if text else None
                except ValueError:
                    raise ValueError(f"{knob.label} must be an integer, got {text!r}") from None
            else:
                values[knob.key] = self.query_one(widget_id, Input).value.strip() or None
        return values

    def _default_tokens(self) -> list[str]:
        """Overrides for recipe-default fields the user actually filled in."""
        if not self._advanced:
            return []
        tokens = []
        for index, (key, _value) in enumerate(self._default_knobs):
            text = self.query_one(f"#param-default-{index}", Input).value.strip()
            if text:
                tokens.append(f"{key}={text}")
        return tokens

    def _env_tokens(self) -> list[str]:
        """env.KEY=value overrides for recipe env fields the user filled in."""
        if not self._advanced:
            return []
        tokens = []
        for index, key in enumerate(self._env_keys):
            text = self.query_one(f"#param-env-{index}", Input).value.strip()
            if text:
                tokens.append(f"env.{key}={text}")
        return tokens

    def _try_launch(self, *, refresh_hosts: bool = False) -> None:
        try:
            tp = self._int_field("param-tp", "Tensor parallel")
            gpu_mem = self._float_field("param-gpu-mem", "GPU memory utilization")
            max_model_len = self._int_field("param-max-len", "Max model len")
            port = self._int_field("param-port", "Serve port")
            knobs = self._knob_values()
            tokens = self._advanced_tokens() + self._default_tokens() + self._env_tokens()
            # Same seam the CLI reaches through _apply_recipe_overrides: value
            # coercion, dotted-key folding, env.* application, runtime resolution.
            recipe, overrides = apply_recipe_overrides(
                tuple(tokens),
                tensor_parallel=tp,
                pipeline_parallel=knobs.get("pipeline_parallel"),
                data_parallel=knobs.get("data_parallel"),
                gpu_mem=gpu_mem,
                max_model_len=max_model_len,
                image=knobs.get("image"),
                # Overrides and planning mutate recipes. A failed or cancelled
                # attempt must not change the defaults for the next attempt.
                recipe=deepcopy(self.recipe),
                port=port,
                served_model_name=knobs.get("served_model_name"),
            )
            executor_config: dict[str, Any] = {}
            if knobs.get("memory_limit"):
                executor_config["memory_limit"] = knobs["memory_limit"]
            if knobs.get("labels"):
                labels = [part for part in str(knobs["labels"]).replace(",", " ").split() if part]
                if any("=" not in label for label in labels):
                    raise ValueError("Container labels must be key=value pairs")
                executor_config["labels"] = labels
            extra_opts: tuple[str, ...] = ()
            if knobs.get("extra_docker_opts"):
                try:
                    extra_opts = tuple(shlex.split(str(knobs["extra_docker_opts"])))
                except ValueError as error:
                    raise ValueError(f"Extra docker opts are malformed: {error}") from None
        except ValueError as error:  # DottedOverrideError is a ValueError subclass
            self._set_error(str(error))
            return
        except Exception as error:  # noqa: BLE001 - keep the dialog open with the reason
            self._set_error(f"Cannot prepare launch: {error}")
            return
        refusal = local_post_commands_refusal(recipe, self.summary.name)
        if refusal is not None and not refresh_hosts and not knobs.get("dry_run"):
            # A dry run executes no hooks, so it stays available for review.
            self._set_error(refusal)
            return
        trust = bool(self.pending_hooks) and self.query_one("#param-trust", Checkbox).value
        if self.pending_hooks and not trust and not refresh_hosts:
            # Without authorization api.run refuses asynchronously and the typed
            # parameters would be lost; keep the dialog open instead.
            self._set_error("Authorize the hooks to run this recipe")
            return
        if self.hosts:
            selected = tuple(host for host, box in self._host_boxes() if box.value)
            # A stale rejection must not prevent checking new parameters or
            # recovered capacity. Recheck the full candidate set, then return
            # to the host choices before launching. Explicitly unchecking all
            # otherwise eligible hosts still requires a selection.
            refresh_hosts = refresh_hosts or (not selected and not self.checks.selectable)
            if refresh_hosts:
                candidates = self.hosts
            else:
                candidates = self._candidate_hosts(selected)
                if not candidates:
                    self._set_error("Select at least one host")
                    return
        else:
            candidates = ()
        options = RunOptions(
            recipe=recipe,
            cluster=self.cluster,
            hosts=candidates or None,
            overrides=overrides,
            trust=trust,
            scheduler=knobs.get("scheduler"),
            cache_dir=knobs.get("cache_dir"),
            workload_id_override=knobs.get("workload_id_override"),
            executor_config=executor_config or None,
            extra_docker_opts=extra_opts or None,
            offline=knobs.get("offline"),
            dry_run=bool(knobs.get("dry_run")),
            ensure=bool(knobs.get("ensure")),
            sync_tuning=knobs.get("sync_tuning", True),
            rootful=bool(knobs.get("rootful")),
            auto_port=bool(knobs.get("auto_port")),
        )
        self._pending_options = options
        self._refreshing_hosts = refresh_hosts
        # Placement from the picker's check is a snapshot: re-query the
        # scheduler with the final options, so a host that filled up while the
        # knobs were edited cannot launch a workload that no longer fits.
        self._set_error("")
        self.app.push_screen(
            RuntimeChecksScreen(
                recipe=self.recipe,
                summary=self.summary,
                cluster=self.cluster,
                hosts=self.hosts,
                sctx=self.sctx,
                options=options,
            ),
            callback=self._preflight_done,
        )

    def _preflight_done(self, checks: RuntimeChecks | None) -> None:
        """Act on the scheduler's answer to a Run press.

        Cancel keeps the typed settings; a clean check launches the pending
        options with the plan the check made from them; anything else refreshes
        the host rows from the new check and stays here, so the user can adjust
        and retry.
        """
        if not self.is_attached:
            return
        options = self._pending_options
        self._pending_options = None
        if checks is None or options is None:
            return
        selected = tuple(host for host, box in self._host_boxes() if box.value)
        if self._refreshing_hosts:
            self._apply_checks(checks, selected or checks.selected)
            self._set_error("" if checks.ok else "; ".join(checks.warnings) or "Scheduler check failed")
            return
        reasons = dict(checks.reasons)
        bad = [host for host in options.hosts or () if host not in checks.selectable]
        if checks.ok and not bad:
            # The plan was built from exactly these options a moment ago, so the
            # launch reuses it: placement is decided once, by this check.
            self.dismiss(RecipeLaunch(options=options, label=self.summary.name, plan=checks.plan))
            return
        self._apply_checks(checks, selected)
        if bad:
            detail = ", ".join(f"{host} ({reasons.get(host, 'insufficient resources')})" for host in bad)
            self._set_error(f"No longer available: {detail}")
            return
        message = "Scheduler check failed"
        if checks.warnings:
            message = f"{message}: {checks.warnings[0]}"
        self._set_error(message)

    def _apply_checks(self, checks: RuntimeChecks, selected: tuple[str, ...]) -> None:
        """Refresh placement, labels, and checkboxes from a new scheduler check."""
        # Boxes the user never changed (or changed back) keep reading as the
        # displayed placement after this refresh; a real edit stays an edit.
        untouched = not self._hosts_edited or frozenset(box_host for box_host, box in self._host_boxes() if box.value) == (
            self._default_selection
        )
        self.checks = checks
        reasons = dict(checks.reasons)
        if self.hosts:
            scheduler = checks.scheduler
            title = f"Hosts ({scheduler})" if scheduler else "Hosts (unavailable)"
            try:
                self.query_one("#param-hosts-title", Static).update(title)
            except Exception:  # noqa: BLE001 - the host section is optional
                pass
        # Not a user edit: the Changed messages these assignments post must not
        # mark the selection as the user's candidate list.
        with self.prevent(Checkbox.Changed):
            for host, box in self._host_boxes():
                eligible = host in checks.selectable
                box.disabled = not eligible
                box.label = Content(host if eligible else f"{host} ({reasons.get(host, 'insufficient resources')})")
                # A disabled row cannot be unchecked by hand, so a host the new
                # check rejected must end unchecked; eligible rows keep the user's
                # own selection.
                box.value = eligible and host in selected
        if untouched:
            self._default_selection = self._eligible_selection(checks, selected)
            self._hosts_edited = False
        self._update_hosts_note()
