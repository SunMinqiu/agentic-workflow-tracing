#!/usr/bin/env python3
"""Replay one bundle under several server configurations and compare them.

A knob sweep needs the workload held fixed. Re-running the agent does not hold
it fixed: the agent's decisions change, so the call count, the prompts and the
tool sequence all differ between arms, and any latency difference mixes the
knob with the workload. Replaying one bundle sends byte-identical requests to
every arm, so the knob is the only thing that changed.

The knobs that matter here (``block_size``, ``cache_dtype``,
``gpu_memory_utilization``, ``enable_prefix_caching``) are engine startup
flags, so an arm means "the server is currently up with that config". This
tool cannot restart the server; it records what the server reports about
itself, so a mislabelled arm is caught rather than believed.

    arm     replay the bundle once against whatever is serving now
    report  tabulate every arm recorded so far
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from agent_io_tracing.analysis.kvcache.summary import PAGE_CSS
from agent_io_tracing.experiments.timezone import (
    EXPERIMENT_TIMEZONE,
    from_epoch,
    now,
)
from agent_io_tracing.replay.runner import replay_to_dir
from agent_io_tracing.serving.vllm_endpoint import parse_cache_config, VLLMEndpoint

WARMING_SPREAD = 0.05

# Startup flags a sweep is likely to move. Any other key that differs between
# arms is reported too; this list only fixes the column order.
KNOBS = (
    "block_size",
    "cache_dtype",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "num_gpu_blocks_override",
    "swap_space",
)


def automatic_label(endpoint: VLLMEndpoint) -> str:
    """Build a unique arm name from the live config and New York time."""
    config = parse_cache_config(endpoint.metrics())
    prefix = str(config.get("enable_prefix_caching", "unknown")).lower()
    prefix = {"true": "on", "false": "off"}.get(prefix, prefix)
    values = (
        f"block{config.get('block_size', 'unknown')}",
        f"dtype-{config.get('cache_dtype', 'unknown')}",
        f"prefix-{prefix}",
        f"gpu{config.get('gpu_memory_utilization', 'unknown')}",
        now().strftime("%Y%m%dT%H%M%S"),
    )
    return "_".join(re.sub(r"[^A-Za-z0-9._-]+", "-", value) for value in values)


def record_arm(
    bundle_path: Path,
    sweep_dir: Path,
    label: str,
    endpoint: VLLMEndpoint,
    mode: str = "packed",
    repeat: int = 1,
    reset_before: bool = True,
) -> list[dict[str, Any]]:
    """Replay the bundle ``repeat`` times into ``sweep_dir/label/rep<i>``.

    The prefix cache is reset before each repetition by default: an arm that
    inherits the previous arm's cache measures the previous arm.
    """
    arm_dir = sweep_dir / label
    if arm_dir.exists():
        raise ValueError(f"arm output already exists: {arm_dir}")
    if repeat < 1:
        raise ValueError("repeat must be positive")
    if not reset_before and repeat != 1:
        raise ValueError("--keep-cache requires --repeat 1")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if reset_before:
        # Fail here rather than after minutes of decoding: vLLM registers
        # /reset_prefix_cache only under VLLM_SERVER_DEV_MODE=1.
        try:
            endpoint.reset_prefix_cache()
        except RuntimeError as exc:
            raise RuntimeError(
                "this server does not expose /reset_prefix_cache; vLLM only "
                "registers it when launched with VLLM_SERVER_DEV_MODE=1. "
                "Relaunch it with that variable, or pass --keep-cache to "
                "accept a warm start. Changing a knob restarts the engine and "
                "empties the cache on its own, so the first repetition of each "
                "arm is cold either way -- but later repetitions are not, and "
                f"the report marks them warm. Underlying error: {exc}"
            ) from exc
    summaries = []
    for index in range(repeat):
        print(
            f"arm: {label}; repetition {index + 1}/{repeat}",
            file=sys.stderr,
            flush=True,
        )
        summary = replay_to_dir(
            json.loads(json.dumps(bundle)),
            endpoint,
            mode,
            sweep_dir / label / f"rep{index}",
            arm={"label": label, "repetition": index},
            reset_before=reset_before,
            bundle_path=bundle_path,
        )
        summaries.append(summary)
    return summaries


def load_arms(sweep_dir: Path) -> list[dict[str, Any]]:
    """One row per arm, medians taken across its repetitions."""
    arms: list[dict[str, Any]] = []
    for arm_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        reps = []
        for rep_dir in sorted(arm_dir.glob("rep*")):
            path = rep_dir / "summary.json"
            if path.is_file():
                reps.append(json.loads(path.read_text(encoding="utf-8")))
        if not reps:
            continue

        def med(key: str, source: list[dict[str, Any]] = reps) -> float | None:
            values = [r.get(key) for r in source]
            values = [v for v in values if isinstance(v, (int, float))]
            return round(statistics.median(values), 3) if values else None

        # Kept in repetition order, not sorted: the question is whether the
        # later passes inherited what the earlier ones cached.
        hit_rates = [
            (r.get("prefix_cache") or {}).get("hit_rate")
            for r in reps
        ]
        hit_rates = [v for v in hit_rates if isinstance(v, (int, float))]
        query_counts = [
            (r.get("prefix_cache") or {}).get("queries") for r in reps
        ]
        query_counts = [v for v in query_counts if isinstance(v, (int, float))]
        hit_counts = [
            (r.get("prefix_cache") or {}).get("hits") for r in reps
        ]
        hit_counts = [v for v in hit_counts if isinstance(v, (int, float))]
        arms.append({
            "label": arm_dir.name,
            "recorded_at": reps[0].get("started_at_epoch_s") or 0.0,
            "reps": len(reps),
            "bundle": Path(reps[0].get("bundle", "")).name,
            "mode": reps[0].get("mode"),
            "requests": reps[0].get("requests"),
            "prompt_tokens": reps[0].get("prompt_tokens"),
            "output_tokens": med("output_tokens"),
            "fixed_output_tokens_per_request": reps[0].get(
                "fixed_output_tokens_per_request"
            ),
            "serving_config": reps[0].get("serving_config") or {},
            "cache_state": ",".join(sorted(
                {str(r.get("cache_state")) for r in reps}
            )),
            "hit_rate": (
                round(statistics.median(hit_rates), 4) if hit_rates else None
            ),
            "hit_rates": hit_rates,
            "cache_queries": round(statistics.median(query_counts)) if query_counts else None,
            "cache_hits": round(statistics.median(hit_counts)) if hit_counts else None,
            "median_ttft_ms": med("median_ttft_ms"),
            "median_tpot_ms": med("median_tpot_ms"),
            "median_total_ms": med("median_total_ms"),
            "wall_s": med("wall_s"),
        })
    # Oldest first: the arm you recorded first is the baseline the rest are
    # measured against, which alphabetical order would scramble.
    return sorted(arms, key=lambda a: (a["recorded_at"], a["label"]))


def comparability(arms: list[dict[str, Any]]) -> list[str]:
    """Reasons these arms are not measuring the same workload."""
    problems = []
    for field, what in (
        ("bundle", "different bundles"),
        ("mode", "different replay modes"),
        ("requests", "different call counts"),
        ("prompt_tokens", "different total input"),
        ("fixed_output_tokens_per_request", "different fixed output lengths"),
        ("cache_state", "different cache states at start"),
    ):
        values = {str(a.get(field)) for a in arms}
        if len(values) > 1:
            problems.append(f"{what}: {', '.join(sorted(values))}")
    return problems


def differing_config_keys(arms: list[dict[str, Any]]) -> list[str]:
    """Serving-config keys that are not identical across arms."""
    keys = {k for a in arms for k in a["serving_config"]}
    differing = {
        k for k in keys
        if len({a["serving_config"].get(k) for a in arms}) > 1
    }
    ordered = [k for k in KNOBS if k in differing]
    return ordered + sorted(differing - set(ordered))


def warming_arms(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Arms whose repetitions are a warming sequence, not replicates.

    Without a cache reset the first pass loads the bundle's prefixes and every
    later pass reads them back, so a median across repetitions mixes a cold
    measurement with a warm one and describes neither.
    """
    return [
        arm for arm in arms
        if len(arm["hit_rates"]) > 1
        and max(arm["hit_rates"]) - min(arm["hit_rates"]) > WARMING_SPREAD
    ]


def _num(value: Any, suffix: str = "", scale: float = 1.0) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value * scale:,.2f}{suffix}"


def _count(value: Any) -> str:
    return f"{int(value):,}" if isinstance(value, (int, float)) else "n/a"


_AUTOMATIC_LABEL = re.compile(
    r"^block(?P<block>[^_]+)_dtype-(?P<dtype>[^_]+)_prefix-(?P<prefix>[^_]+)_"
    r"gpu(?P<gpu>[^_]+)_(?P<stamp>\d{8}T\d{6})(?P<utc>Z?)$"
)


def pretty_arm(arm: dict[str, Any]) -> str:
    """Format an arm like a task name while keeping its directory unchanged."""
    label = str(arm["label"])
    match = _AUTOMATIC_LABEL.match(label)
    if match:
        values = match.groupdict()
        stamp_time = _dt.datetime.strptime(values["stamp"], "%Y%m%dT%H%M%S")
        if values["utc"]:
            stamp_time = stamp_time.replace(
                tzinfo=_dt.timezone.utc
            ).astimezone(EXPERIMENT_TIMEZONE)
        stamp = stamp_time.strftime("%m-%d %H:%M")
        try:
            gpu = f"{float(values['gpu']):.2f}"
        except ValueError:
            gpu = values["gpu"]
        return (
            f"block {values['block']} · {values['dtype']} · "
            f"prefix {values['prefix']} · GPU {gpu} · {stamp}"
        )
    recorded_at = arm.get("recorded_at")
    if isinstance(recorded_at, (int, float)) and recorded_at > 0:
        stamp = from_epoch(recorded_at).strftime("%m-%d %H:%M")
        return f"{label} · {stamp}"
    return label


def render(arms: list[dict[str, Any]]) -> list[str]:
    if not arms:
        return ["No arms recorded yet."]
    config_keys = differing_config_keys(arms)
    headers = config_keys + [
        "median TTFT", "median TPOT", "median request latency", "wall",
        "vs baseline",
    ]
    warming = warming_arms(arms)
    baseline = arms[0].get("median_ttft_ms")
    lines = [
        "| arm | " + " | ".join(headers) + " |",
        "| --- |" + " --- |" * len(config_keys)
        + " ---: |" * (len(headers) - len(config_keys)),
    ]
    for arm in arms:
        ttft = arm.get("median_ttft_ms")
        if arm is arms[0]:
            delta = "baseline"
        elif isinstance(ttft, (int, float)) and baseline:
            delta = f"{(ttft / baseline - 1) * 100:+.2f}% TTFT"
        else:
            delta = "n/a"
        lines.append("| " + " | ".join(
            [pretty_arm(arm)]
            + [str(arm["serving_config"].get(k, "?")) for k in config_keys]
            + [
                _num(arm.get("median_ttft_ms"), "s", 0.001),
                _num(arm.get("median_tpot_ms"), "ms/token"),
                _num(arm.get("median_total_ms"), "s", 0.001),
                _num(arm.get("wall_s"), "s"),
                delta,
            ]
        ) + " |")
    fixed_output = arms[0].get("fixed_output_tokens_per_request")
    output_description = (
        f"output fixed at {_count(fixed_output)} tokens per request"
        if isinstance(fixed_output, (int, float))
        else "legacy output length"
    )
    lines.append("")
    lines.append(
        f"{arms[0]['requests']} calls, "
        f"{arms[0]['prompt_tokens']:,} total input tokens, "
        f"{output_description}, "
        f"replayed in `{arms[0]['mode']}` mode from `{arms[0]['bundle']}`."
    )
    for arm in warming:
        lines.append("")
        lines.append(
            f"**{arm['label']}: its repetitions are not replicates.** Hit rate "
            f"went {arm['hit_rates'][0]:.2%} → {arm['hit_rates'][-1]:.2%} "
            "across them, so the first pass loaded the prefixes the later ones "
            "read back. Every median on this row mixes a cold pass with a warm "
            "one and describes neither; take the repetitions apart, or give "
            "the server VLLM_SERVER_DEV_MODE=1 so each pass starts cold."
        )
    if any("cold_since_restart" in a["cache_state"] for a in arms):
        lines.append("")
        lines.append(
            "`cold*` = the engine's query counter was still at zero, so it had "
            "served nothing since restart."
        )
    problems = comparability(arms)
    if problems:
        lines.append("")
        lines.append("**Not comparable** — " + "; ".join(problems) + ".")
    if len(arms) == 1:
        lines.append("")
        lines.append("One arm only — nothing to compare against yet.")
    elif not config_keys:
        lines.append("")
        lines.append(
            "Every arm reports the same serving config; the server was not "
            "restarted with a different knob between them, so any difference "
            "above is run-to-run noise, not a knob effect."
        )
    return lines


def _token_summary(arms: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| arm | calls | total input | total output | cache queries | cache hits | realized reuse | repetitions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in arms:
        rate = _num(arm.get("hit_rate"), "%", 100)
        lines.append(
            f"| {pretty_arm(arm)} | {arm['requests']} | {arm['prompt_tokens']:,} | "
            f"{_count(arm.get('output_tokens'))} | "
            f"{_count(arm.get('cache_queries'))} | {_count(arm.get('cache_hits'))} | "
            f"{rate} | {arm['reps']} |"
        )
    return lines


def _write_comparison_plot(sweep_dir: Path, arms: list[dict[str, Any]]) -> Path | None:
    if not arms:
        return None
    import matplotlib.pyplot as plt

    labels = [arm["label"] for arm in arms]
    positions = list(range(len(labels)))
    height = max(3.5, 0.7 * len(labels) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, height))
    rates = [
        arm["hit_rate"] * 100 if isinstance(arm.get("hit_rate"), (int, float)) else 0
        for arm in arms
    ]
    ttft = [
        arm["median_ttft_ms"] / 1000 if isinstance(arm.get("median_ttft_ms"), (int, float)) else 0
        for arm in arms
    ]
    axes[0].barh(positions, rates, color="#4c78a8")
    axes[0].set_xlabel("Realized reuse (%)")
    axes[0].set_yticks(positions, labels=labels)
    axes[0].invert_yaxis()
    axes[1].barh(positions, ttft, color="#f58518")
    axes[1].set_xlabel("Median TTFT (s)")
    axes[1].set_yticks(positions, labels=[])
    axes[1].invert_yaxis()
    fig.suptitle(f"Fixed-input test: {sweep_dir.name}")
    fig.tight_layout()
    viz = sweep_dir / "visualizations"
    viz.mkdir(parents=True, exist_ok=True)
    path = viz / "sweep_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _per_arm_sections(sweep_dir: Path, arms: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for arm in arms:
        lines.extend([f"### {pretty_arm(arm)}", ""])
        lines.append(
            f"- Bundle: `{arm['bundle']}` · mode: `{arm['mode']}` · "
            f"cache: `{arm['cache_state']}` · repetitions: {arm['reps']}"
        )
        lines.extend(["", "**Server config**", ""])
        lines.extend(["| key | value |", "| --- | --- |"])
        for key, value in sorted(arm["serving_config"].items()):
            lines.append(f"| `{key}` | `{value}` |")
        lines.extend(["", "**Artifacts**", ""])
        for index in range(arm["reps"]):
            base = f"{arm['label']}/rep{index}"
            lines.append(
                f"- repetition {index + 1}: [summary]({base}/summary.json) · "
                f"[requests]({base}/requests.jsonl) · "
                f"[metrics before]({base}/metrics_before.prom) · "
                f"[metrics after]({base}/metrics_after.prom)"
            )
        lines.extend(["", "---", ""])
    return lines


def write_report(sweep_dir: Path, results_dir: Path | None = None) -> Path:
    import markdown

    arms = load_arms(sweep_dir)
    comparison_plot = _write_comparison_plot(sweep_dir, arms)
    lines = [
        f"# Fixed-input Test — {sweep_dir.name}",
        "",
        f"_Generated {_dt.date.today().isoformat()} · {len(arms)} arm(s) · prefill-only replay._",
        "",
        "## Summary — tokens",
        "",
    ]
    lines.extend(_token_summary(arms))
    lines.extend(["", "## Summary — time and configuration", ""])
    lines.extend(render(arms))
    if comparison_plot is not None:
        lines.extend(["", "![Realized reuse and TTFT by arm](visualizations/sweep_comparison.png)", ""])
    lines.extend(["", "## Per arm", ""])
    lines.extend(_per_arm_sections(sweep_dir, arms))
    if results_dir is not None:
        index = results_dir.resolve() / "index.html"
        lines.extend([f"[Back to experiments index]({os.path.relpath(index, sweep_dir.resolve())})", ""])
    text = "\n".join(lines) + "\n"
    (sweep_dir / "sweep.md").write_text(text, encoding="utf-8")
    (sweep_dir / "kvcache_report.md").write_text(text, encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables"])
    html_text = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Fixed-input Test — {sweep_dir.name}</title>\n<style>\n{PAGE_CSS}"
        f"</style>\n</head>\n<body>\n{body}"
        + "\n</body>\n</html>\n"
    )
    path = sweep_dir / "kvcache_report.html"
    path.write_text(html_text, encoding="utf-8")
    (sweep_dir / "sweep.html").write_text(html_text, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    arm = sub.add_parser("arm", help="replay the bundle against the live server")
    arm.add_argument("--bundle", type=Path, required=True)
    arm.add_argument("--sweep-dir", type=Path, required=True)
    arm.add_argument(
        "--label",
        help="Optional result name; defaults to live config plus New York time.",
    )
    arm.add_argument("--mode", choices=("packed", "paced"), default="packed")
    arm.add_argument("--repeat", type=int, default=1)
    arm.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not reset the prefix cache first; only for a warm-start arm.",
    )
    arm.add_argument(
        "--url",
        default=(
            os.environ.get("VLLM_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        ),
    )
    arm.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY") or "")
    arm.add_argument("--results", type=Path, default=Path("results"))

    report = sub.add_parser("report", help="tabulate the arms recorded so far")
    report.add_argument("--sweep-dir", type=Path, required=True)
    report.add_argument("--results", type=Path, default=Path("results"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "report":
        if not args.sweep_dir.is_dir():
            raise SystemExit(f"{args.sweep_dir} does not exist")
        path = write_report(args.sweep_dir, args.results)
        from agent_io_tracing.analysis.results_index import write_results_index
        index_path = write_results_index(args.results.resolve())
        print("\n".join(render(load_arms(args.sweep_dir))))
        print(f"\nwrote {path}")
        print(f"index → {index_path}")
        return

    if not args.url:
        raise SystemExit("Set VLLM_URL or pass --url.")
    endpoint = VLLMEndpoint(args.url, args.api_key, timeout_s=300.0)
    label = args.label or "automatic"
    try:
        if not args.label:
            label = automatic_label(endpoint)
        summaries = record_arm(
            args.bundle,
            args.sweep_dir,
            label,
            endpoint,
            mode=args.mode,
            repeat=args.repeat,
            reset_before=not args.keep_cache,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"arm {label} failed: {exc}") from exc
    write_report(args.sweep_dir, args.results)
    from agent_io_tracing.analysis.results_index import write_results_index
    index_path = write_results_index(args.results.resolve())
    print(f"index → {index_path}", file=sys.stderr)
    print(json.dumps(summaries[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
