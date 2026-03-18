#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
from queue import Queue
import sys
import time
from typing import Any

from core.buffer import AdvancedDataBuffer
from core.config import CONFIG, initialize_runtime_config
from core.history import TuningHistory
from doctor import collect_doctor_checks, print_doctor_report, summarize_doctor_checks
from llm.client import LLMTuner
from pid_safety import (
    apply_pid_guardrails,
    build_fallback_suggestion,
    is_good_enough,
    maybe_update_best_result,
    pid_equals,
    should_rollback_to_best,
)
from sim.model import HeatingSimulator, SETPOINT
from sim.runtime import (
    EVENT_DECISION,
    EVENT_LIFECYCLE,
    EVENT_ROLLBACK,
    EVENT_ROUND_METRICS,
    EVENT_SAMPLE,
    QueueEventSink,
    SimulationController,
    now_elapsed,
    publish_event,
    wait_while_paused,
)
from system_id import extract_initial_pid, system_identify


TUI_DEPENDENCIES = ("textual",)


def ensure_runtime_config(
    verbose: bool = False, create_if_missing: bool = True
) -> None:
    initialize_runtime_config(create_if_missing=create_if_missing, verbose=verbose)


ensure_runtime_config(verbose=False, create_if_missing=False)


def choose_tui_language(default: str = "zh") -> str:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return default

    print("Choose interface language / 选择界面语言")
    print("[1] 中文")
    print("[2] English")
    choice = input("Press Enter for 中文 / 回车默认中文: ").strip().lower()
    if choice in {"2", "en", "english"}:
        return "en"
    return default


def _console(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def _emit_lifecycle(
    event_sink: QueueEventSink | None,
    start_time: float,
    phase: str,
    message: str,
) -> None:
    publish_event(
        event_sink,
        EVENT_LIFECYCLE,
        phase=phase,
        message=message,
        elapsed_sec=now_elapsed(start_time),
    )


def _publish_doctor_checks(
    doctor_checks: list[Any] | None,
    event_sink: QueueEventSink | None = None,
    emit_console: bool = True,
) -> None:
    if not doctor_checks:
        return

    summary = summarize_doctor_checks(doctor_checks)
    _console(emit_console, f"[Doctor] {summary}")
    publish_event(
        event_sink,
        EVENT_LIFECYCLE,
        phase="doctor",
        message=summary,
        elapsed_sec=0.0,
    )
    for check in doctor_checks:
        if getattr(check, "status", "PASS") == "PASS":
            continue
        message = f"{check.name}: {check.detail}"
        _console(emit_console, f"[Doctor:{check.status}] {message}")
        publish_event(
            event_sink,
            EVENT_LIFECYCLE,
            phase=f"doctor_{str(check.status).lower()}",
            message=message,
            elapsed_sec=0.0,
        )


def _run_simulator_warm_start(
    sim: HeatingSimulator,
    event_sink: QueueEventSink | None = None,
    emit_console: bool = True,
) -> dict[str, float] | None:
    probe = HeatingSimulator(random_seed=0)
    probe.set_pid(0.0, 0.0, 0.0)
    time_data: list[float] = []
    temp_data: list[float] = []
    pwm_data: list[float] = []

    sample_count = max(40, min(80, int(CONFIG.get("BUFFER_SIZE", 100))))
    for _ in range(sample_count):
        probe.pwm = 255.0
        probe.update()
        data = probe.get_data()
        time_data.append(float(data["timestamp"]))
        temp_data.append(float(data["input"]))
        pwm_data.append(float(data["pwm"]))

    result = system_identify(time_data, temp_data, pwm_data)
    candidate_pid = extract_initial_pid(result, "PID")
    if not candidate_pid:
        message = "Warm start skipped because system identification did not return a usable PID."
        _console(emit_console, f"[Warm Start] {message}")
        publish_event(
            event_sink,
            EVENT_LIFECYCLE,
            phase="warm_start",
            message=message,
            elapsed_sec=0.0,
        )
        return None

    safe_pid, notes = apply_pid_guardrails(
        {"p": sim.kp, "i": sim.ki, "d": sim.kd},
        candidate_pid,
    )
    sim.set_pid(safe_pid["p"], safe_pid["i"], safe_pid["d"])
    note_text = f" ({'; '.join(notes)})" if notes else ""
    message = (
        f"Applied warm start PID: P={safe_pid['p']:.4f} "
        f"I={safe_pid['i']:.4f} D={safe_pid['d']:.4f}{note_text}"
    )
    _console(emit_console, f"[Warm Start] {message}")
    publish_event(
        event_sink,
        EVENT_LIFECYCLE,
        phase="warm_start",
        message=message,
        elapsed_sec=0.0,
    )
    return safe_pid


def _collect_data(
    sim: Any,
    buffer: AdvancedDataBuffer,
    event_sink: QueueEventSink | None = None,
    controller: SimulationController | None = None,
) -> tuple[int, bool]:
    steps = 0
    max_simulink_run_steps = 200
    simulink_run_count = 0

    while not buffer.is_full():
        if not wait_while_paused(controller):
            return steps, False
        if controller is not None and controller.should_stop:
            return steps, False

        if hasattr(sim, "compute_pid"):
            sim.compute_pid()
            sim.update()
            data = sim.get_data()
            buffer.add(data)
            publish_event(
                event_sink,
                EVENT_SAMPLE,
                timestamp=float(data.get("timestamp", 0.0)),
                setpoint=float(data.get("setpoint", 0.0)),
                input=float(data.get("input", 0.0)),
                pwm=float(data.get("pwm", 0.0)),
                error=float(data.get("error", 0.0)),
                p=float(data.get("p", 0.0)),
                i=float(data.get("i", 0.0)),
                d=float(data.get("d", 0.0)),
            )
            steps += 1
            continue

        simulink_run_count += 1
        if simulink_run_count > max_simulink_run_steps:
            raise RuntimeError(
                "Simulink data collection timed out before filling the buffer."
            )

        sim.run_step()
        for data in sim.get_data():
            if controller is not None and controller.should_stop:
                return steps, False
            buffer.add(data)
            publish_event(
                event_sink,
                EVENT_SAMPLE,
                timestamp=float(data.get("timestamp", 0.0)),
                setpoint=float(data.get("setpoint", 0.0)),
                input=float(data.get("input", 0.0)),
                pwm=float(data.get("pwm", 0.0)),
                error=float(data.get("error", 0.0)),
                p=float(data.get("p", 0.0)),
                i=float(data.get("i", 0.0)),
                d=float(data.get("d", 0.0)),
            )
            steps += 1
            if buffer.is_full():
                break

    return steps, True


def _run_tuning_loop(
    sim: Any,
    setpoint: float,
    mode_label: str,
    event_sink: QueueEventSink | None = None,
    controller: SimulationController | None = None,
    emit_console: bool = True,
    warm_start: bool = True,
    doctor_checks: list[Any] | None = None,
) -> dict[str, Any]:
    tuner = LLMTuner(
        CONFIG["LLM_API_KEY"],
        CONFIG["LLM_API_BASE_URL"],
        CONFIG["LLM_MODEL_NAME"],
        CONFIG["LLM_PROVIDER"],
    )
    buffer = AdvancedDataBuffer(max_size=CONFIG["BUFFER_SIZE"])
    history = TuningHistory(max_history=5)
    good_enough_rules = {
        "avg_error_threshold": CONFIG["GOOD_ENOUGH_AVG_ERROR"],
        "steady_state_error_threshold": CONFIG["GOOD_ENOUGH_STEADY_STATE_ERROR"],
        "overshoot_threshold": CONFIG["GOOD_ENOUGH_OVERSHOOT"],
    }

    round_num = 0
    last_round = 0
    stable_rounds = 0
    best_result = None
    last_metrics: dict[str, Any] = {}
    start_time = time.time()
    completed_reason = "max_rounds_reached"
    fallback_count = 0
    guardrail_count = 0
    rollback_count = 0

    _publish_doctor_checks(
        doctor_checks,
        event_sink=event_sink,
        emit_console=emit_console,
    )

    if warm_start and isinstance(sim, HeatingSimulator):
        _run_simulator_warm_start(sim, event_sink=event_sink, emit_console=emit_console)

    buffer.current_pid = {"p": sim.kp, "i": sim.ki, "d": sim.kd}
    buffer.setpoint = setpoint
    _emit_lifecycle(event_sink, start_time, "starting", f"{mode_label} simulation started.")

    try:
        while round_num < CONFIG["MAX_TUNING_ROUNDS"]:
            if controller is not None and controller.should_stop:
                completed_reason = "stopped_by_user"
                _console(emit_console, "\n[INFO] Simulation stopped by user.")
                _emit_lifecycle(event_sink, start_time, "stopped", "Simulation stopped by user.")
                break

            round_index = round_num + 1
            _console(emit_console, f"\n[Round {round_index}] Collecting data...")
            _emit_lifecycle(
                event_sink,
                start_time,
                "collecting",
                f"Collecting data for round {round_index}.",
            )

            steps, completed = _collect_data(
                sim,
                buffer,
                event_sink=event_sink,
                controller=controller,
            )
            if not completed:
                completed_reason = "stopped_by_user"
                _console(emit_console, "\n[INFO] Simulation stopped by user.")
                _emit_lifecycle(event_sink, start_time, "stopped", "Simulation stopped by user.")
                break

            last_round = round_index
            _console(emit_console, f"[Round {round_index}] Collected {steps} samples.")

            metrics = buffer.calculate_advanced_metrics()
            last_metrics = dict(metrics)
            current_pid = {"p": sim.kp, "i": sim.ki, "d": sim.kd}
            stable_rounds = (
                stable_rounds + 1
                if is_good_enough(metrics, good_enough_rules)
                else 0
            )
            publish_event(
                event_sink,
                EVENT_ROUND_METRICS,
                round=round_index,
                avg_error=float(metrics["avg_error"]),
                max_error=float(metrics["max_error"]),
                steady_state_error=float(metrics["steady_state_error"]),
                overshoot=float(metrics["overshoot"]),
                zero_crossings=int(metrics["zero_crossings"]),
                status=str(metrics["status"]),
                stable_rounds=stable_rounds,
            )

            previous_best = best_result
            best_result = maybe_update_best_result(
                best_result, current_pid, metrics, round_index
            )
            if best_result is not None and best_result is not previous_best:
                _emit_lifecycle(
                    event_sink,
                    start_time,
                    "best_result",
                    f"Captured a new best stable result at round {round_index}.",
                )

            if (
                best_result
                and not pid_equals(current_pid, best_result["pid"])
                and should_rollback_to_best(metrics, best_result["metrics"])
            ):
                sim.set_pid(
                    best_result["pid"]["p"],
                    best_result["pid"]["i"],
                    best_result["pid"]["d"],
                )
                buffer.current_pid = dict(best_result["pid"])
                publish_event(
                    event_sink,
                    EVENT_ROLLBACK,
                    round=round_index,
                    target_round=int(best_result["round"]),
                    pid=dict(best_result["pid"]),
                    reason="Current metrics regressed against the best stable result.",
                )
                rollback_count += 1
                if is_good_enough(best_result["metrics"], good_enough_rules):
                    completed_reason = "rollback_to_best"
                    _emit_lifecycle(
                        event_sink,
                        start_time,
                        "completed",
                        "Rolled back to the best stable result and finished early.",
                    )
                    break
                round_num += 1
                buffer.reset()
                continue

            if (
                metrics["avg_error"] < CONFIG["MIN_ERROR_THRESHOLD"]
                and metrics["status"] == "STABLE"
            ):
                completed_reason = "low_error_converged"
                _emit_lifecycle(
                    event_sink,
                    start_time,
                    "completed",
                    "Simulation converged with stable low error.",
                )
                break

            if stable_rounds >= CONFIG["REQUIRED_STABLE_ROUNDS"]:
                completed_reason = "stable_rounds_reached"
                _emit_lifecycle(
                    event_sink,
                    start_time,
                    "completed",
                    f"Reached {stable_rounds} stable rounds and finished early.",
                )
                break

            _emit_lifecycle(
                event_sink,
                start_time,
                "llm_request",
                f"Requesting PID suggestion for round {round_index}.",
            )
            result = tuner.analyze(buffer.to_prompt_data(), history.to_prompt_text())

            if not result:
                _emit_lifecycle(
                    event_sink,
                    start_time,
                    "fallback",
                    f"LLM unavailable at round {round_index}; using fallback rules.",
                )
                result = build_fallback_suggestion(buffer.current_pid, metrics)

            analysis = str(
                result.get("analysis_summary", "No analysis summary was provided.")
            )
            thought = str(result.get("thought_process", ""))
            action = str(result.get("tuning_action", "UNKNOWN"))
            safe_pid, guardrail_notes = apply_pid_guardrails(current_pid, result)
            sim.set_pid(safe_pid["p"], safe_pid["i"], safe_pid["d"])
            history.add_record(round_index, safe_pid, metrics, analysis, thought)
            buffer.current_pid = dict(safe_pid)
            publish_event(
                event_sink,
                EVENT_DECISION,
                round=round_index,
                action=action,
                analysis_summary=analysis,
                fallback_used=bool(result.get("fallback_used")),
                guardrail_notes=list(guardrail_notes),
            )
            if result.get("fallback_used"):
                fallback_count += 1
            if guardrail_notes:
                guardrail_count += 1

            round_num += 1
            buffer.reset()

            if result.get("status") == "DONE":
                completed_reason = "llm_marked_done"
                _emit_lifecycle(
                    event_sink,
                    start_time,
                    "completed",
                    "LLM marked the tuning run as done.",
                )
                break

    except KeyboardInterrupt:
        completed_reason = "keyboard_interrupt"
        _console(emit_console, "\n[INFO] Simulation interrupted by keyboard.")
        _emit_lifecycle(
            event_sink,
            start_time,
            "stopped",
            "Simulation interrupted by keyboard.",
        )
    except Exception as exc:
        completed_reason = "error"
        _console(emit_console, f"\n[ERROR] Simulation failed: {exc}")
        _emit_lifecycle(
            event_sink,
            start_time,
            "error",
            f"Simulation failed: {exc}",
        )
        raise
    finally:
        elapsed_sec = now_elapsed(start_time)
        _emit_lifecycle(
            event_sink,
            start_time,
            "finished",
            f"Simulation finished in {elapsed_sec:.1f}s.",
        )
        _console(
            emit_console,
            (
                f"\n[Summary] elapsed={elapsed_sec:.1f}s "
                f"final_pid=P={sim.kp:.4f} I={sim.ki:.4f} D={sim.kd:.4f}"
            ),
        )

    return {
        "provider": CONFIG["LLM_PROVIDER"],
        "model": CONFIG["LLM_MODEL_NAME"],
        "rounds_completed": last_round,
        "elapsed_sec": now_elapsed(start_time),
        "final_pid": {"p": sim.kp, "i": sim.ki, "d": sim.kd},
        "final_metrics": dict(last_metrics),
        "stopped": bool(controller.should_stop) if controller is not None else False,
        "fallback_count": fallback_count,
        "guardrail_count": guardrail_count,
        "rollback_count": rollback_count,
        "completed_reason": completed_reason,
    }


def determine_tui_mode(force_plain: bool, matlab_model_path: str) -> tuple[bool, str | None]:
    if force_plain:
        return False, None

    if matlab_model_path:
        return False, "Simulink mode does not support the TUI yet; falling back to plain output."

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False, "The TUI requires an interactive terminal; falling back to plain output."

    missing = [
        dependency
        for dependency in TUI_DEPENDENCIES
        if importlib.util.find_spec(dependency) is None
    ]
    if missing:
        return (
            False,
            "TUI dependencies are missing: "
            + ", ".join(missing)
            + ". Falling back to plain output.",
        )

    return True, None


def _run_python_simulation_with_tui(
    warm_start: bool = True,
    doctor_checks: list[Any] | None = None,
) -> dict[str, Any]:
    from sim.tui import SimulationTUIApp

    event_queue: Queue[dict[str, Any]] = Queue()
    controller = SimulationController()
    event_sink = QueueEventSink(event_queue)
    sim = HeatingSimulator()
    result_box: dict[str, Any] = {}
    language = choose_tui_language()

    def worker() -> None:
        result_box["result"] = _run_tuning_loop(
            sim,
            SETPOINT,
            "Python",
            event_sink=event_sink,
            controller=controller,
            emit_console=False,
            warm_start=warm_start,
            doctor_checks=doctor_checks,
        )

    SimulationTUIApp(
        event_queue=event_queue,
        controller=controller,
        worker_target=worker,
        event_sink=event_sink,
        mode_label="Python",
        language=language,
    ).run()
    return result_box.get("result", {})


def _run_python_simulation_plain(
    warm_start: bool = True,
    doctor_checks: list[Any] | None = None,
) -> dict[str, Any]:
    print("=" * 60)
    print("  LLM PID Tuner PRO - Simulation")
    print("=" * 60)
    print(f"Setpoint: {SETPOINT}, Model: {CONFIG['LLM_MODEL_NAME']}")
    sim = HeatingSimulator()
    return _run_tuning_loop(
        sim,
        SETPOINT,
        "Python",
        emit_console=True,
        warm_start=warm_start,
        doctor_checks=doctor_checks,
    )


def _run_simulink_simulation() -> dict[str, Any] | None:
    try:
        from sim.simulink_bridge import SimulinkBridge
    except ImportError as exc:
        print(f"[ERROR] {exc}")
        return None

    matlab_model_path = CONFIG.get("MATLAB_MODEL_PATH", "").strip()
    pid_block_path = CONFIG.get("MATLAB_PID_BLOCK_PATH", "").strip()
    output_signal = CONFIG.get("MATLAB_OUTPUT_SIGNAL", "").strip()

    try:
        sim_step_time = float(CONFIG.get("MATLAB_SIM_STEP_TIME", 10.0))
        setpoint = float(CONFIG.get("MATLAB_SETPOINT", 200.0))
    except (TypeError, ValueError) as exc:
        print(f"[ERROR] Invalid Simulink numeric configuration: {exc}")
        return None

    if not pid_block_path:
        print("[ERROR] MATLAB_PID_BLOCK_PATH is required for Simulink mode.")
        return None
    if not output_signal:
        print("[ERROR] MATLAB_OUTPUT_SIGNAL is required for Simulink mode.")
        return None

    print("=" * 60)
    print("  LLM PID Tuner PRO - Simulink")
    print("=" * 60)
    print(f"Setpoint: {setpoint}, Model: {CONFIG['LLM_MODEL_NAME']}")
    print(f"Simulink model: {matlab_model_path}")

    sim = SimulinkBridge(
        model_path=matlab_model_path,
        setpoint=setpoint,
        pid_block_path=pid_block_path,
        output_signal=output_signal,
        sim_step_time=sim_step_time,
    )

    try:
        sim.connect()
    except Exception as exc:
        print(f"[ERROR] Failed to connect to Simulink: {exc}")
        return None

    try:
        return _run_tuning_loop(sim, setpoint, "Simulink", emit_console=True)
    finally:
        sim.disconnect()


def run_simulation(force_plain: bool = False) -> dict[str, Any] | None:
    ensure_runtime_config(verbose=True)
    doctor_checks = collect_doctor_checks()
    matlab_model_path = CONFIG.get("MATLAB_MODEL_PATH", "").strip()
    use_tui, fallback_message = determine_tui_mode(force_plain, matlab_model_path)

    if fallback_message:
        print(f"[WARN] {fallback_message}")

    if matlab_model_path:
        print_doctor_report(doctor_checks)
        return _run_simulink_simulation()

    if use_tui:
        try:
            return _run_python_simulation_with_tui(
                warm_start=True,
                doctor_checks=doctor_checks,
            )
        except ImportError as exc:
            print(f"[WARN] Failed to start the TUI ({exc}); falling back to plain output.")

    print_doctor_report(doctor_checks)
    return _run_python_simulation_plain(
        warm_start=True,
        doctor_checks=doctor_checks,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the LLM PID simulator.")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable the Textual dashboard and use plain console logs.",
    )
    args = parser.parse_args(argv)
    run_simulation(force_plain=args.plain)


if __name__ == "__main__":
    main()
