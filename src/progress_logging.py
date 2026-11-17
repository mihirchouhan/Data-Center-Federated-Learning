"""
Mandatory Terminal Progress Logging and Live Tracking Module.

Provides real-time terminal progress reporting and persistent logging
for Federated Learning (FedAvg), Centralized Training, and Evaluation runs.

Features:
- Zero external dependencies (tqdm-compatible pure Python progress bar).
- High-visibility terminal output with immediate flushing (VS Code Remote SSH safe).
- Unbuffered dual-destination logging (terminal stdout + persistent progress.log).
- Real-time client progress and local epoch reporting.
- Genuine, non-fabricated ETA calculation with rolling-window smoothing.
- Structured CSV round logging (round_progress.csv).
- Graceful interrupt handling (flushes logs and preserves checkpoints on SIGINT).
"""
from __future__ import annotations

import csv
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Formatting and Timing Utilities
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    if seconds is None or math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "00:00:00"
    total_sec = int(round(seconds))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def calculate_eta(
    elapsed_sec: float,
    completed_steps: int,
    total_steps: int,
    recent_durations: Optional[Sequence[float]] = None,
    window_size: int = 10,
) -> float:
    """
    Calculate genuine, un-fabricated ETA in seconds.

    If recent_durations is provided, uses rolling average of recent steps
    to adapt to non-stationary step durations; otherwise projects from total elapsed.
    """
    if completed_steps <= 0 or total_steps <= completed_steps:
        return 0.0

    remaining_steps = total_steps - completed_steps

    if recent_durations and len(recent_durations) > 0:
        window = recent_durations[-window_size:]
        avg_step_sec = sum(window) / len(window)
    else:
        avg_step_sec = elapsed_sec / completed_steps

    return max(0.0, remaining_steps * avg_step_sec)


# ---------------------------------------------------------------------------
# Dual Logger (Stdout + File with Immediate Flush)
# ---------------------------------------------------------------------------

class DualLogger:
    """
    Logger that writes simultaneously to sys.stdout and a persistent log file.
    All writes are immediately flushed and fsynced.
    """

    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
        prefix: str = "",
        auto_timestamp: bool = False,
    ):
        self.prefix = prefix
        self.auto_timestamp = auto_timestamp
        self.file_handle = None
        self.log_path = None

        if log_path is not None:
            self.log_path = Path(log_path).resolve()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_handle = open(self.log_path, "a", encoding="utf-8", buffering=1)

    def write_line(self, line: str, to_terminal: bool = True, to_file: bool = True) -> None:
        """Write a complete line to terminal and/or log file."""
        ts = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " if self.auto_timestamp else ""
        formatted = f"{ts}{line}"

        if to_terminal:
            print(formatted, flush=True)

        if to_file and self.file_handle is not None and not self.file_handle.closed:
            self.file_handle.write(formatted + "\n")
            self.file_handle.flush()
            try:
                os.fsync(self.file_handle.fileno())
            except OSError:
                pass

    def terminal_inplace(self, text: str) -> None:
        """Write dynamic updating text to terminal via carriage return (no file write)."""
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        if is_tty:
            sys.stdout.write(f"\r\033[K{text}")
            sys.stdout.flush()
        else:
            # In non-interactive/piped environments, throttle and line-break cleanly
            now = time.monotonic()
            if not hasattr(self, "_last_non_tty_time"):
                self._last_non_tty_time = 0.0
            if (now - self._last_non_tty_time >= 0.4) or ("100.0%" in text) or ("Clients " in text and text.endswith("100.0%")):
                sys.stdout.write(f"{text}\n")
                sys.stdout.flush()
                self._last_non_tty_time = now

    def terminal_clear_line(self) -> None:
        """Clear current terminal line."""
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        if is_tty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def close(self) -> None:
        """Close log file handle cleanly."""
        if self.file_handle is not None and not self.file_handle.closed:
            self.file_handle.flush()
            try:
                os.fsync(self.file_handle.fileno())
            except OSError:
                pass
            self.file_handle.close()
            self.file_handle = None

    def __enter__(self) -> "DualLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Zero-Dependency Terminal Progress Bar (tqdm-equivalent)
# ---------------------------------------------------------------------------

class TerminalProgressBar:
    """
    Lightweight, high-performance terminal progress bar with ETA and rate tracking.
    Works smoothly over SSH, PTYs, and standard Linux shells without dependencies.
    """

    def __init__(
        self,
        iterable: Optional[Iterable[Any]] = None,
        total: Optional[int] = None,
        desc: str = "Progress",
        unit: str = "it",
        bar_len: int = 25,
        min_interval_sec: float = 0.08,  # Max ~12 updates/sec to prevent SSH lag
        ascii_only: bool = False,
    ):
        self.iterable = iterable
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            try:
                total = len(iterable)  # type: ignore
            except TypeError:
                total = None
        self.total = total or 0
        self.desc = desc
        self.unit = unit
        self.bar_len = bar_len
        self.min_interval_sec = min_interval_sec
        self.ascii_only = ascii_only

        self.n = 0
        self.start_time = time.monotonic()
        self.last_update_time = self.start_time
        self.closed = False

        # Glyphs
        self.filled_char = "#" if ascii_only else "█"
        self.empty_char = "-" if ascii_only else "░"

    def __iter__(self) -> Iterator[Any]:
        if self.iterable is None:
            raise ValueError("No iterable provided to ProgressBar")
        self.start_time = time.monotonic()
        self.last_update_time = self.start_time
        self.render()
        for item in self.iterable:
            yield item
            self.update(1)
        self.close()

    def update(self, n: int = 1) -> None:
        """Increment progress by n units."""
        self.n += n
        now = time.monotonic()
        if (now - self.last_update_time >= self.min_interval_sec) or (self.total and self.n >= self.total):
            self.render()
            self.last_update_time = now

    def render(self) -> None:
        """Render the progress bar line to terminal."""
        if self.closed:
            return

        now = time.monotonic()
        elapsed = now - self.start_time
        rate = (self.n / elapsed) if elapsed > 0 else 0.0

        if self.total > 0:
            pct = min(1.0, max(0.0, self.n / self.total))
            filled_len = int(round(self.bar_len * pct))
            bar = (self.filled_char * filled_len) + (self.empty_char * (self.bar_len - filled_len))
            pct_str = f"{pct * 100.0:5.1f}%"
            eta_sec = calculate_eta(elapsed, self.n, self.total)
            eta_str = format_time(eta_sec)
            elapsed_str = format_time(elapsed)

            line = (
                f"\r{self.desc}: [{bar}] {self.n}/{self.total} ({pct_str}) | "
                f"{rate:6.1f} {self.unit}/s | Elapsed: {elapsed_str} | ETA: {eta_str}"
            )
        else:
            elapsed_str = format_time(elapsed)
            line = f"\r{self.desc}: {self.n} {self.unit} | {rate:6.1f} {self.unit}/s | Elapsed: {elapsed_str}"

        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        if is_tty:
            sys.stdout.write(f"\r\033[K{line}")
        else:
            # Throttle non-TTY updates and use newlines
            if not hasattr(self, "_last_render_time"):
                self._last_render_time = 0.0
            if (now - self._last_render_time >= 0.5) or (self.total and self.n >= self.total):
                sys.stdout.write(f"{line}\n")
                self._last_render_time = now
        sys.stdout.flush()

    def close(self) -> None:
        """Cleanly terminate the progress bar with a newline."""
        if not self.closed:
            self.render()
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.closed = True

    def __enter__(self) -> "TerminalProgressBar":
        self.start_time = time.monotonic()
        self.last_update_time = self.start_time
        self.render()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# FedAvg Live Progress Logger
# ---------------------------------------------------------------------------

class FedAvgProgressLogger:
    """
    Mandatory Progress Logger for Federated Learning runs.

    Produces real-time terminal output adhering to the exact specification:
    [Experiment | Config | Seed]
    Round 001/200 | Clients 098/098
    Local Epoch 1/1 | Loss: 0.18342
    Round completed | Val MAE: 31.82 W
    Time: 12.4s | Elapsed: 00:00:12 | ETA: 00:41:08

    Simultaneously records clean milestones into progress.log and round_progress.csv.
    """

    def __init__(
        self,
        experiment_name: str,
        config_name: str,
        seed: int,
        total_rounds: int,
        total_clients: int,
        device: str = "cpu",
        output_dir: Optional[Union[str, Path]] = None,
        log_filename: str = "progress.log",
        csv_filename: str = "round_progress.csv",
        interactive: bool = True,
    ):
        self.experiment_name = experiment_name
        self.config_name = config_name
        self.seed = seed
        self.total_rounds = total_rounds
        self.total_clients = total_clients
        self.device = str(device)
        self.interactive = interactive

        self.output_dir = Path(output_dir).resolve() if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.output_dir / log_filename
            self.csv_path = self.output_dir / csv_filename
        else:
            self.log_path = None
            self.csv_path = None

        self.dual_logger = DualLogger(self.log_path)

        # Timing tracking
        self.experiment_start_time = time.monotonic()
        self.current_round_start_time = time.monotonic()
        self.recent_round_durations: List[float] = []

        # State
        self.current_round = 0
        self.best_val_mae = float("inf")
        self.best_val_rmse = float("inf")
        self.best_round = 0

        # Initialize CSV log if path provided
        if self.csv_path:
            write_header = not self.csv_path.exists()
            self.csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            if write_header:
                self.csv_writer.writerow([
                    "round",
                    "total_rounds",
                    "clients_sampled",
                    "train_loss",
                    "val_mae_w",
                    "val_rmse_w",
                    "is_best",
                    "round_time_sec",
                    "elapsed_sec",
                    "eta_sec",
                    "timestamp",
                ])
                self.csv_file.flush()
        else:
            self.csv_file = None
            self.csv_writer = None

        # Install graceful interrupt handler
        self._setup_interrupt_handler()

    def _setup_interrupt_handler(self) -> None:
        def handle_interrupt(signum, frame):
            msg = (
                f"\n[INTERRUPT] Received signal {signum}. Training interrupted at Round "
                f"{self.current_round}/{self.total_rounds}. Flushed progress and logs."
            )
            self.dual_logger.write_line(msg)
            self.close()
            sys.exit(130)

        try:
            signal.signal(signal.SIGINT, handle_interrupt)
            signal.signal(signal.SIGTERM, handle_interrupt)
        except (ValueError, AttributeError):
            pass

    def start_training(self) -> None:
        """Log experiment start header."""
        header = (
            f"================================================================================\n"
            f"🚀 STARTING FEDERATED TRAINING: [{self.experiment_name} | {self.config_name} | Seed {self.seed}]\n"
            f"Device: {self.device} | Total Rounds: {self.total_rounds} | Clients/Round: {self.total_clients}\n"
            f"Started At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"================================================================================"
        )
        self.dual_logger.write_line(header)
        self.experiment_start_time = time.monotonic()

    def start_round(self, round_num: int) -> None:
        """Signal start of a new FedAvg round."""
        self.current_round = round_num
        self.current_round_start_time = time.monotonic()

    def update_client_progress(
        self,
        client_idx: int,
        epoch: int = 1,
        total_epochs: int = 1,
        loss: Optional[float] = None,
    ) -> None:
        """
        Display dynamic live client progress on terminal.
        Does not pollute log files; updates in-place via carriage return.
        """
        if not self.interactive:
            return

        pct = (client_idx / self.total_clients) * 100.0 if self.total_clients > 0 else 0.0
        bar_len = 15
        filled = int(round(bar_len * (pct / 100.0)))
        bar = ("█" * filled) + ("░" * (bar_len - filled))
        loss_str = f"{loss:.5f}" if loss is not None else "..."

        status = (
            f"\r[{self.experiment_name} | {self.config_name} | Seed {self.seed}] "
            f"Round {self.current_round:03d}/{self.total_rounds:03d} | "
            f"Clients {client_idx:03d}/{self.total_clients:03d} [{bar}] {pct:4.1f}% | "
            f"Epoch {epoch}/{total_epochs} | Loss: {loss_str}"
        )
        self.dual_logger.terminal_inplace(status)

    def complete_round(
        self,
        train_loss: float,
        val_mae: Optional[float] = None,
        val_rmse: Optional[float] = None,
        total_epochs: int = 1,
        extra_info: str = "",
    ) -> None:
        """
        Record and display a completed round.
        Prints the exact required 5-line specification to terminal and persists to progress.log.
        """
        now = time.monotonic()
        round_duration = now - self.current_round_start_time
        elapsed_total = now - self.experiment_start_time
        self.recent_round_durations.append(round_duration)

        eta_total = calculate_eta(
            elapsed_sec=elapsed_total,
            completed_steps=self.current_round,
            total_steps=self.total_rounds,
            recent_durations=self.recent_round_durations,
        )

        is_best = False
        if val_mae is not None:
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_val_rmse = val_rmse if val_rmse is not None else float("nan")
                self.best_round = self.current_round
                is_best = True

        # Clear any in-place terminal text
        if self.interactive:
            self.dual_logger.terminal_clear_line()

        # Build output block matching exact specification
        val_mae_str = f"{val_mae:.2f} W" if val_mae is not None else "N/A"
        if val_rmse is not None and is_best:
            val_extra = f" (Best: R{self.best_round:03d} {self.best_val_mae:.2f} W)"
        else:
            val_extra = ""
        if extra_info:
            val_extra += f" | {extra_info}"

        block = (
            f"[{self.experiment_name} | {self.config_name} | Seed {self.seed}]\n"
            f"Round {self.current_round:03d}/{self.total_rounds:03d} | Clients {self.total_clients:03d}/{self.total_clients:03d}\n"
            f"Local Epoch {total_epochs}/{total_epochs} | Loss: {train_loss:.5f}\n"
            f"Round completed | Val MAE: {val_mae_str}{val_extra}\n"
            f"Time: {round_duration:.1f}s | Elapsed: {format_time(elapsed_total)} | ETA: {format_time(eta_total)}"
        )
        self.dual_logger.write_line(block)

        # Write to structured CSV
        if self.csv_writer is not None and self.csv_file is not None and not self.csv_file.closed:
            self.csv_writer.writerow([
                self.current_round,
                self.total_rounds,
                self.total_clients,
                round(train_loss, 6),
                round(val_mae, 4) if val_mae is not None else "",
                round(val_rmse, 4) if val_rmse is not None else "",
                int(is_best),
                round(round_duration, 2),
                round(elapsed_total, 2),
                round(eta_total, 2),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ])
            self.csv_file.flush()
            try:
                os.fsync(self.csv_file.fileno())
            except OSError:
                pass

    def finish_training(self) -> None:
        """Log training completion summary."""
        elapsed_total = time.monotonic() - self.experiment_start_time
        avg_round = (elapsed_total / self.total_rounds) if self.total_rounds > 0 else 0.0
        summary = (
            f"================================================================================\n"
            f"✅ FEDERATED TRAINING COMPLETE: [{self.experiment_name} | {self.config_name} | Seed {self.seed}]\n"
            f"Rounds: {self.current_round}/{self.total_rounds} | Best Val Round: R{self.best_round:03d} (MAE: {self.best_val_mae:.2f} W)\n"
            f"Total Time: {format_time(elapsed_total)} ({elapsed_total:.1f}s) | Avg Round: {avg_round:.1f}s\n"
            f"Completed At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"================================================================================"
        )
        self.dual_logger.write_line(summary)
        self.close()

    def close(self) -> None:
        """Close log handles."""
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.flush()
            try:
                os.fsync(self.csv_file.fileno())
            except OSError:
                pass
            self.csv_file.close()
            self.csv_file = None
        self.dual_logger.close()


# ---------------------------------------------------------------------------
# Evaluation Live Progress Logger
# ---------------------------------------------------------------------------

class EvaluationProgressLogger:
    """
    Mandatory Progress Logger for Evaluation runs.

    Displays:
    - Dataset name and total window count.
    - Model/configuration evaluated.
    - Real-time progress bar over nodes/windows with throughput rate and ETA.
    - Final MAE/RMSE and throughput upon completion.
    """

    def __init__(
        self,
        dataset_name: str,
        total_windows: int,
        total_nodes: int,
        model_name: str = "Model",
        output_dir: Optional[Union[str, Path]] = None,
        log_filename: str = "progress.log",
        interactive: bool = True,
    ):
        self.dataset_name = dataset_name
        self.total_windows = total_windows
        self.total_nodes = total_nodes
        self.model_name = model_name
        self.interactive = interactive

        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.log_path = (self.output_dir / log_filename) if self.output_dir else None
        self.dual_logger = DualLogger(self.log_path)

        self.start_time = time.monotonic()
        self.nodes_completed = 0
        self.windows_completed = 0

    def start_evaluation(self) -> None:
        """Log evaluation start banner."""
        banner = (
            f"\n--- [Evaluation | {self.dataset_name}] Evaluating {self.model_name} ---\n"
            f"Total Windows: {self.total_windows:,} across {self.total_nodes} nodes | Started: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.dual_logger.write_line(banner)
        self.start_time = time.monotonic()

    def update_node(self, node_idx: int, windows_in_node: int) -> None:
        """Update progress after processing a node."""
        self.nodes_completed += 1
        self.windows_completed += windows_in_node

        if not self.interactive:
            return

        now = time.monotonic()
        elapsed = now - self.start_time
        rate = (self.windows_completed / elapsed) if elapsed > 0 else 0.0
        eta = calculate_eta(elapsed, self.windows_completed, self.total_windows)

        pct = (self.windows_completed / self.total_windows * 100.0) if self.total_windows > 0 else 0.0
        bar_len = 20
        filled = int(round(bar_len * (pct / 100.0)))
        bar = ("█" * filled) + ("░" * (bar_len - filled))

        status = (
            f"\rEvaluating: [{bar}] {self.nodes_completed}/{self.total_nodes} nodes "
            f"({pct:5.1f}%) | {self.windows_completed:,}/{self.total_windows:,} win | "
            f"Rate: {rate:6.0f} win/s | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}"
        )
        self.dual_logger.terminal_inplace(status)

    def complete_evaluation(
        self,
        mae: float,
        rmse: float,
        macro_mae: Optional[float] = None,
        macro_rmse: Optional[float] = None,
        nodes_beating_pers: Optional[int] = None,
    ) -> None:
        """Display and record evaluation summary."""
        now = time.monotonic()
        elapsed = now - self.start_time
        rate = (self.windows_completed / elapsed) if elapsed > 0 else 0.0

        if self.interactive:
            self.dual_logger.terminal_clear_line()

        macro_str = ""
        if macro_mae is not None and macro_rmse is not None:
            macro_str = f" | Macro MAE: {macro_mae:.2f} W | Macro RMSE: {macro_rmse:.2f} W"

        node_pers_str = ""
        if nodes_beating_pers is not None:
            pct_beat = (nodes_beating_pers / self.total_nodes * 100.0) if self.total_nodes > 0 else 0.0
            node_pers_str = f" | Beating Pers: {nodes_beating_pers}/{self.total_nodes} ({pct_beat:.1f}%)"

        line = (
            f"[Evaluation Complete | {self.dataset_name}]\n"
            f"Model: {self.model_name} | Windows: {self.windows_completed:,} ({self.nodes_completed} nodes)\n"
            f"Pooled MAE: {mae:.2f} W | Pooled RMSE: {rmse:.2f} W{macro_str}{node_pers_str}\n"
            f"Time: {elapsed:.1f}s | Throughput: {rate:,.0f} windows/s"
        )
        self.dual_logger.write_line(line)
        self.dual_logger.close()


# ---------------------------------------------------------------------------
# Experiment Stage and Orchestration Tracker
# ---------------------------------------------------------------------------

class ExperimentStageTracker:
    """
    Tracks overall experiment suite progression:
    - Experiment index (e.g., [Experiment 1/3]).
    - Active configuration and seed.
    - Stage transitions (Preprocessing, Training, Validation, Testing, Reporting).
    - Immediate error and warning alerts.
    """

    def __init__(
        self,
        suite_name: str,
        total_experiments: int = 1,
        output_dir: Optional[Union[str, Path]] = None,
        log_filename: str = "progress.log",
    ):
        self.suite_name = suite_name
        self.total_experiments = total_experiments
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.log_path = (self.output_dir / log_filename) if self.output_dir else None
        self.dual_logger = DualLogger(self.log_path, auto_timestamp=False)

        self.current_experiment = 0
        self.current_config = ""
        self.current_seed = 0
        self.current_stage = ""
        self.stage_start_time = time.monotonic()
        self.suite_start_time = time.monotonic()

    def set_experiment(self, exp_idx: int, config_name: str, seed: int) -> None:
        """Set active experiment configuration and seed."""
        self.current_experiment = exp_idx
        self.current_config = config_name
        self.current_seed = seed
        banner = (
            f"\n{'#' * 80}\n"
            f"[Experiment {exp_idx}/{self.total_experiments}] "
            f"Suite: {self.suite_name} | Config: {config_name} | Seed: {seed}\n"
            f"{'#' * 80}"
        )
        self.dual_logger.write_line(banner)

    def set_stage(self, stage_name: str, details: str = "") -> None:
        """Record stage transition."""
        self.current_stage = stage_name.upper()
        self.stage_start_time = time.monotonic()
        det_str = f" - {details}" if details else ""
        line = f"\n>>> STAGE: {self.current_stage}{det_str} [{datetime.now().strftime('%H:%M:%S')}]"
        self.dual_logger.write_line(line)

    def log_warning(self, message: str) -> None:
        """Log immediate warning alert."""
        line = f"⚠️ [WARNING] {message} [{datetime.now().strftime('%H:%M:%S')}]"
        self.dual_logger.write_line(line)

    def log_error(self, message: str, exc: Optional[Exception] = None) -> None:
        """Log immediate error alert."""
        line = f"❌ [ERROR] {message} [{datetime.now().strftime('%H:%M:%S')}]"
        if exc is not None:
            line += f" | Exception: {type(exc).__name__}: {str(exc)}"
        self.dual_logger.write_line(line)

    def finish_stage(self) -> None:
        """Mark stage completion."""
        elapsed = time.monotonic() - self.stage_start_time
        line = f"<<< STAGE COMPLETED: {self.current_stage} in {elapsed:.1f}s"
        self.dual_logger.write_line(line)

    def finish_suite(self) -> None:
        """Mark overall suite completion."""
        elapsed = time.monotonic() - self.suite_start_time
        banner = (
            f"\n{'=' * 80}\n"
            f"🏆 SUITE COMPLETED: {self.suite_name} ({self.total_experiments} experiments)\n"
            f"Total Elapsed Time: {format_time(elapsed)} ({elapsed:.1f}s)\n"
            f"{'=' * 80}"
        )
        self.dual_logger.write_line(banner)
        self.dual_logger.close()
