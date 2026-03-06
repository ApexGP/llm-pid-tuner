#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM PID 调参基准脚本。

目标：
1. 用固定随机种子复现实验；
2. 对比 baseline / fallback / real-llm 三种路径；
3. 输出简洁、可比较的指标结果。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List

import simulator
from pid_safety import (
    DEFAULT_CONVERGENCE_RULES,
    apply_pid_guardrails,
    build_fallback_suggestion,
    is_good_enough,
    maybe_update_best_result,
    pid_equals,
    should_rollback_to_best,
)
from tuner import AdvancedDataBuffer, LLMTuner, TuningHistory


DEFAULT_CASES = ("baseline", "fallback", "llm")


def create_llm_tuner() -> LLMTuner:
    api_base_url = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_API_KEY", "")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")
    provider = os.getenv("LLM_PROVIDER", "openai")

    if not api_key:
        raise RuntimeError("未设置 LLM_API_KEY，无法运行 llm benchmark")

    return LLMTuner(api_key, api_base_url, model_name, provider)


def run_case(case_name: str, rounds: int, seed: int, stop_on_done: bool = True) -> Dict[str, Any]:
    random.seed(seed)
    simulator.random.seed(seed)
    simulator.kp, simulator.ki, simulator.kd = 1.0, 0.1, 0.05

    sim = simulator.HeatingSimulator()
    history = TuningHistory(max_history=5)
    llm = create_llm_tuner() if case_name == "llm" else None
    fallback_count = 0
    best_result = None
    records: List[Dict[str, Any]] = []
    start_time = time.time()

    for round_num in range(1, rounds + 1):
        buffer = AdvancedDataBuffer(max_size=simulator.BUFFER_SIZE)
        buffer.current_pid = {"p": simulator.kp, "i": simulator.ki, "d": simulator.kd}
        buffer.setpoint = simulator.SETPOINT

        while not buffer.is_full():
            sim.compute_pid()
            sim.update()
            buffer.add(sim.get_data())

        metrics = buffer.calculate_advanced_metrics()
        record = {
            "round": round_num,
            "avg_error": metrics["avg_error"],
            "steady_state_error": metrics["steady_state_error"],
            "overshoot": metrics["overshoot"],
            "status": metrics["status"],
            "pid": {"p": simulator.kp, "i": simulator.ki, "d": simulator.kd},
        }
        records.append(record)

        current_pid = {"p": simulator.kp, "i": simulator.ki, "d": simulator.kd}
        best_result = maybe_update_best_result(best_result, current_pid, metrics, round_num)

        if best_result and not pid_equals(current_pid, best_result["pid"]) and should_rollback_to_best(metrics, best_result["metrics"]):
            simulator.kp = best_result["pid"]["p"]
            simulator.ki = best_result["pid"]["i"]
            simulator.kd = best_result["pid"]["d"]
            if is_good_enough(best_result["metrics"], DEFAULT_CONVERGENCE_RULES):
                break
            continue

        if case_name == "baseline":
            continue

        if case_name == "fallback":
            result = build_fallback_suggestion(buffer.current_pid, metrics)
        else:
            result = llm.analyze(buffer.to_prompt_data(), history.to_prompt_text())
            if not result:
                result = build_fallback_suggestion(buffer.current_pid, metrics)

        if result.get("fallback_used"):
            fallback_count += 1

        safe_pid, _ = apply_pid_guardrails(buffer.current_pid, result)
        simulator.kp, simulator.ki, simulator.kd = safe_pid["p"], safe_pid["i"], safe_pid["d"]
        history.add_record(round_num, safe_pid, metrics, result.get("analysis_summary", ""))

        if stop_on_done and result.get("status") == "DONE":
            break

    elapsed = time.time() - start_time
    final_metrics = records[-1]

    return {
        "case": case_name,
        "rounds_executed": len(records),
        "fallback_count": fallback_count,
        "elapsed_sec": elapsed,
        "final": {
            "avg_error": final_metrics["avg_error"],
            "steady_state_error": final_metrics["steady_state_error"],
            "overshoot": final_metrics["overshoot"],
            "status": final_metrics["status"],
            "pid": {"p": simulator.kp, "i": simulator.ki, "d": simulator.kd},
        },
        "history": records,
    }


def print_summary(results: List[Dict[str, Any]]):
    print("=" * 88)
    print("LLM PID Benchmark Summary")
    print("=" * 88)
    for result in results:
        final = result["final"]
        pid = final["pid"]
        print(
            f"{result['case']:<10} rounds={result['rounds_executed']:<2} "
            f"avg_err={final['avg_error']:.3f} steady_err={final['steady_state_error']:.3f} "
            f"overshoot={final['overshoot']:.2f}% status={final['status']:<12} "
            f"pid=({pid['p']:.4f},{pid['i']:.4f},{pid['d']:.4f}) "
            f"fallbacks={result['fallback_count']:<2} elapsed={result['elapsed_sec']:.1f}s"
        )


def main():
    parser = argparse.ArgumentParser(description="LLM PID 调参 benchmark")
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=list(DEFAULT_CASES), help="要运行的 benchmark case")
    parser.add_argument("--rounds", type=int, default=8, help="每个 case 最多运行的轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-stop-on-done", action="store_true", help="即使模型判定 DONE 也继续跑满轮数")
    parser.add_argument("--json-out", type=str, help="将结果写入 JSON 文件")
    args = parser.parse_args()

    results = [
        run_case(case_name, rounds=args.rounds, seed=args.seed, stop_on_done=not args.no_stop_on_done)
        for case_name in args.cases
    ]

    print_summary(results)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 结果已写入 {args.json_out}")


if __name__ == "__main__":
    main()
