"""Tests for sparkrun.core.progress — unified logging & progress system."""

from __future__ import annotations

import logging

from sparkrun.core.progress import (
    PROGRESS,
    VERBOSE,
    LaunchProgress,
    Verbosity,
    PHASE_LABELS,
    TOTAL_PHASES,
)


class TestCustomLevels:
    """Verify custom log levels are registered correctly."""

    def test_progress_level_value(self):
        assert PROGRESS == 25

    def test_verbose_level_value(self):
        assert VERBOSE == 15

    def test_progress_level_name(self):
        assert logging.getLevelName(PROGRESS) == "PROGRESS"

    def test_verbose_level_name(self):
        assert logging.getLevelName(VERBOSE) == "VERBOSE"

    def test_progress_between_info_and_warning(self):
        assert logging.INFO < PROGRESS < logging.WARNING

    def test_verbose_between_debug_and_info(self):
        assert logging.DEBUG < VERBOSE < logging.INFO


class TestVerbosity:
    """Verify Verbosity enum values."""

    def test_default(self):
        assert Verbosity.DEFAULT == 0

    def test_detail(self):
        assert Verbosity.DETAIL == 1

    def test_verbose(self):
        assert Verbosity.VERBOSE == 2

    def test_debug(self):
        assert Verbosity.DEBUG == 3


class TestPhaseLabels:
    """Verify fixed phase definitions."""

    def test_total_phases(self):
        assert TOTAL_PHASES == 6

    def test_all_phases_labeled(self):
        for i in range(1, TOTAL_PHASES + 1):
            assert i in PHASE_LABELS

    def test_phase_labels_content(self):
        assert PHASE_LABELS[1] == "Preparing"
        assert PHASE_LABELS[2] == "Building image"
        assert PHASE_LABELS[3] == "Distributing resources"
        assert PHASE_LABELS[4] == "Syncing tuning configs"
        assert PHASE_LABELS[5] == "Launching runtime"
        assert PHASE_LABELS[6] == "Post-launch hooks"


class TestLaunchProgressPhases:
    """Test phase tracking API."""

    def test_phase_emits_progress_log(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase(1)
        assert "[1/6] Preparing" in caplog.text

    def test_phase_custom_label(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase(1, "Custom label")
        assert "[1/6] Custom label" in caplog.text

    def test_phase_end_emits_done(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase(1)
            p.phase_end(elapsed=1.5)
        assert "done (1.5s)" in caplog.text

    def test_phase_end_auto_elapsed(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase(1)
            p.phase_end()
        assert "done (" in caplog.text

    def test_phase_skip(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase_skip(2, "no builder")
        assert "[2/6] Building image" in caplog.text
        assert "skipped (no builder)" in caplog.text

    def test_phase_skip_no_reason(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase_skip(2)
        assert "skipped" in caplog.text
        # No parenthesized reason
        assert "()" not in caplog.text

    def test_auto_close_phase_on_new_phase(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase(1)
            p.phase(2)  # should auto-close phase 1
        assert "done (" in caplog.text
        assert "[2/6]" in caplog.text


class TestLaunchProgressSteps:
    """Test step tracking API."""

    def test_step_with_total(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase(5)
            p.begin_runtime_steps(3)
            p.step("Cleaning up")
        assert "Step 1/3: Cleaning up" in caplog.text

    def test_step_increments(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase(5)
            p.begin_runtime_steps(3)
            p.step("First")
            p.step("Second")
        assert "Step 1/3: First" in caplog.text
        assert "Step 2/3: Second" in caplog.text

    def test_step_without_total(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase(5)
            p.step("Something")
        assert "Step 1: Something" in caplog.text

    def test_step_done_emits_detail(self, caplog):
        p = LaunchProgress(Verbosity.DETAIL)
        with caplog.at_level(logging.INFO, logger="sparkrun.progress"):
            p.phase(5)
            p.begin_runtime_steps(1)
            import time

            t0 = p.step("Test")
            time.sleep(0.01)
            p.step_done(t0)
        assert "step done (" in caplog.text

    def test_step_resets_on_new_phase(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase(5)
            p.begin_runtime_steps(3)
            p.step("First")
            p.phase_end()
            p.phase(6)
            p.begin_runtime_steps(2)
            p.step("New first")
        assert "Step 1/2: New first" in caplog.text


class TestLaunchProgressTieredOutput:
    """Test detail/verbose/debug/warn/error methods."""

    def test_detail_visible_at_detail(self, caplog):
        p = LaunchProgress(Verbosity.DETAIL)
        with caplog.at_level(logging.INFO, logger="sparkrun.progress"):
            p.detail("detail message %s", "here")
        assert "detail message here" in caplog.text

    def test_detail_hidden_at_default(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.detail("should not appear")
        assert "should not appear" not in caplog.text

    def test_verbose_visible_at_verbose(self, caplog):
        p = LaunchProgress(Verbosity.VERBOSE)
        with caplog.at_level(VERBOSE, logger="sparkrun.progress"):
            p.verbose("verbose msg")
        assert "verbose msg" in caplog.text

    def test_verbose_hidden_at_detail(self, caplog):
        p = LaunchProgress(Verbosity.DETAIL)
        with caplog.at_level(logging.INFO, logger="sparkrun.progress"):
            p.verbose("should not appear")
        assert "should not appear" not in caplog.text

    def test_debug_visible_at_debug(self, caplog):
        p = LaunchProgress(Verbosity.DEBUG)
        with caplog.at_level(logging.DEBUG, logger="sparkrun.progress"):
            p.debug("debug msg")
        assert "debug msg" in caplog.text

    def test_warn_always_visible(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.warn("warning msg")
        assert "warning msg" in caplog.text

    def test_error_always_visible(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.error("error msg")
        assert "error msg" in caplog.text


class TestLaunchProgressInit:
    """Test initialization."""

    def test_default_verbosity(self):
        p = LaunchProgress()
        assert p.verbosity == Verbosity.DEFAULT

    def test_custom_verbosity(self):
        p = LaunchProgress(Verbosity.DEBUG)
        assert p.verbosity == Verbosity.DEBUG

    def test_logger_name(self):
        p = LaunchProgress()
        assert p._log.name == "sparkrun.progress"


class TestFullPipeline:
    """Integration test: simulate a full launch pipeline's progress output."""

    def test_full_pipeline_default_verbosity(self, caplog):
        p = LaunchProgress(Verbosity.DEFAULT)
        with caplog.at_level(PROGRESS, logger="sparkrun.progress"):
            p.phase(1)
            p.phase_end(elapsed=0.3)

            p.phase_skip(2, "no builder")

            p.phase(3)
            p.detail("Distribution mode: local")  # hidden at default
            p.phase_end(elapsed=45.2)

            p.phase(4)
            p.phase_end(elapsed=1.1)

            p.phase(5)
            p.begin_runtime_steps(7)
            p.step("Cleaning up existing containers")
            p.step("Detecting InfiniBand")
            p.step("Detecting head node IP")
            p.step("Launching containers")
            p.step("Running pre-serve hooks")
            p.step("Starting head node serve")
            p.step("Starting worker nodes")
            p.phase_end(elapsed=28.4)

            p.phase_skip(6)

        text = caplog.text
        # All phases visible
        assert "[1/6] Preparing" in text
        assert "[2/6] Building image" in text
        assert "[3/6] Distributing resources" in text
        assert "[4/6] Syncing tuning configs" in text
        assert "[5/6] Launching runtime" in text
        assert "[6/6] Post-launch hooks" in text

        # Steps visible
        assert "Step 1/7: Cleaning up existing containers" in text
        assert "Step 7/7: Starting worker nodes" in text

        # Detail hidden at default
        assert "Distribution mode: local" not in text

        # Skipped phases
        assert "skipped (no builder)" in text
