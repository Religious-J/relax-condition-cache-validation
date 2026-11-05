"""Recompute selected historical measurements; never submit or run training."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from benchmark_condition_cache import compare_runs, read_events, summarize_run
from verify import ROOT, verify


def read_json(path: Path) -> dict:
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(data)


def recorded_numerical(report: dict) -> dict:
    """Inspect stored metrics, not the unavailable original gradient tensors."""
    if not report.get("passed") or not report.get("cache_branch_verified") or report.get("relative_tolerance") != 0:
        raise ValueError("Stored numerical report lacks exact actual-cache coverage")
    loss = report["loss"]
    if not loss["passed"] or not loss["finite"] or loss["absolute_error"] != 0:
        raise ValueError("Stored loss report is not finite/exact")
    counts = {}
    for section in ("gradients", "parameter_updates"):
        values = report[section]
        if not values:
            raise ValueError(f"Empty stored {section}")
        counts[section] = len(values)
        for row in values.values():
            numbers = [row[k] for k in ("reference_l2", "actual_l2", "error_l2", "max_absolute_error")]
            if not all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in numbers):
                raise ValueError(f"Invalid stored {section} norm/error")
            if not row["finite"] or not row["passed"] or row["error_l2"] != 0 or row["max_absolute_error"] != 0:
                raise ValueError(f"Stored {section} is not exact")
            if row["reference_l2"] == 0 and row["actual_l2"] != 0:
                raise ValueError("Nonzero stored actual for zero reference")
    if not any(row["reference_l2"] > 0 for row in report["parameter_updates"].values()):
        raise ValueError("Stored FP32 updates are all zero")
    return {
        "stored_report_exact": True,
        "tensor_counts": counts,
        "max_absolute_error": 0,
        "loss_absolute_error": loss["absolute_error"],
        "update_quantity": report["update_quantity"],
        "cache_validation": report["cache_validation"],
        "raw_tensor_comparisons_recomputed": False,
    }


def observed_resources(samples: list[dict], job: dict) -> dict:
    selected = [
        s
        for s in samples
        if s.get("stage") == "sample" and job["start_time"] / 1000 <= s["unix"] <= job["end_time"] / 1000
    ]

    def total(values: list) -> int | None:
        return sum(values) if all(type(v) in (int, float) for v in values) else None

    fields = {
        "ray_process_rss_sum_bytes": lambda s: total(
            [v.get("VmRSS_bytes") for v in s.get("namespace_ray_process_memory", {}).values()]
        ),
        "gpu_compute_app_memory_sum_bytes": lambda s: total(
            [v.get("gpu_used_memory_bytes") for v in s.get("gpu_apps", [])]
        ),
        "host_shm_used_bytes": lambda s: s.get("shm_used_bytes"),
        "cache_allocated_block_bytes": lambda s: total(
            [s.get("cache", {}).get(k, {}).get("physical_bytes") for k in ("payload", "pending", "metadata")]
        ),
    }
    result = {
        "sample_count": len(selected),
        "scope": "15-second samples within Ray job start/end; not instantaneous peaks",
    }
    for key, extract in fields.items():
        values = [extract(row) for row in selected]
        known = [v for v in values if v is not None]
        result[key] = {"maximum": max(known) if known else None, "observed": len(known)}
    return result


def lifecycle(events: list[dict], after: dict) -> dict:
    closes = [e for e in events if e["stage"] == "close"]
    steps = [e for e in events if e["stage"] == "step"]
    fields = {}
    for key in ("pending_copy_count", "pending_copy_bytes", "h2d_wait_seconds", "cache_errors"):
        rows = steps + closes if key.startswith("pending_") else steps
        known = [e[key] for e in rows if key in e]
        fields[key] = {"observed": len(known), "expected": len(rows), "maximum": max(known) if known else None}
    return {
        "normal_close_records": len(closes),
        "all_recorded_closes_complete": all(e.get("status") == "complete" for e in closes),
        "boundary_counters": fields,
        "post_job": after,
        "scope": "Normal completion only; does not substitute for cancellation/fault injection",
    }


def cache_cost(events: list[dict], cold_steps: int) -> dict | None:
    if not cold_steps:
        return None
    per_rank = defaultdict(dict)
    for event in events:
        if event["stage"] == "step":
            per_rank[event["rank"]][event["step"]] = event.get("cache_stats", {})
    fields = {}
    for field in ("lock_wait_seconds", "lock_hold_seconds", "checksum_seconds", "map_seconds", "load_seconds"):
        values = []
        for steps in per_rank.values():
            before, after = steps[cold_steps - 1], steps[max(steps)]
            if field in before and field in after:
                values.append(after[field] - before[field])
        if values:
            fields[field] = {"mean_per_rank": statistics.mean(values), "min": min(values), "max": max(values)}
    return (
        {
            "scope": "Cumulative cache-stat differences: last cold step to last training step. Prefetch crosses boundaries; these are overlapping host diagnostics, not critical-path time or additive speedup.",
            "counters": fields,
        }
        if fields
        else None
    )


def recompute() -> dict:
    integrity = verify()
    manifest = read_json(ROOT / "MANIFEST.json")
    result = {"integrity": integrity, "campaigns": {}}
    with tempfile.TemporaryDirectory(prefix="no17-public-recompute-") as scratch:
        for campaign in manifest["campaigns"]:
            base = ROOT / "evidence" / campaign["id"]
            samples = [
                json.loads(line) for line in gzip.decompress((base / "resources.jsonl.gz").read_bytes()).splitlines()
            ]
            runs, groups, timings = {}, defaultdict(lambda: {"online": [], "cached": []}), {}
            for row in campaign["runs"]:
                name = row["name"]
                published = base / "runs" / name
                directory = Path(scratch) / campaign["id"] / name
                directory.mkdir(parents=True)
                for path in published.glob("events-rank*.jsonl.gz"):
                    (directory / path.stem).write_bytes(gzip.decompress(path.read_bytes()))
                (directory / "job.json").write_bytes((published / "job.json").read_bytes())
                measurement = summarize_run(directory, cold_steps=row["cold_steps"])
                measurement["path"] = str(published.relative_to(ROOT))
                if not measurement["complete"] or not measurement["job_complete"]:
                    raise ValueError(f"Incomplete stored run {name}: {measurement['issues']}")
                events = read_events(directory)
                step_times = defaultdict(list)
                for event in events:
                    if event["stage"] == "step":
                        step_times[event["step"]].append(event["wall_seconds"])
                timings[name] = [max(step_times[i]) for i in sorted(step_times)]
                runs[name] = {
                    "measurement": measurement,
                    "resources": observed_resources(samples, read_json(published / "job.json")),
                    "lifecycle": lifecycle(events, read_json(published / "resource_after.json")),
                    "cache_cost": cache_cost(events, row["cold_steps"]),
                }
                numerical = published / "conditioning-verification.json.gz"
                if row["purpose"] == "verify":
                    runs[name]["numerical"] = recorded_numerical(read_json(numerical))
                else:
                    groups[row["group"]][measurement["mode"]].append(measurement)
            comparisons = {}
            for name, modes in groups.items():
                compared = compare_runs(modes["online"], modes["cached"])
                if not compared["comparable"]:
                    raise ValueError(f"Incomparable group {name}: {compared['reasons']}")
                pairs = []
                if len(modes["online"]) != len(modes["cached"]):
                    raise ValueError(f"Unpaired run count: {name}")
                for online, cached in zip(modes["online"], modes["cached"]):
                    a, b = Path(online["path"]).name, Path(cached["path"]).name
                    accumulated, gaps = 0.0, []
                    if len(timings[a]) != len(timings[b]):
                        raise ValueError(f"Unpaired step count: {name}")
                    for ta, tb in zip(timings[a], timings[b]):
                        accumulated += tb - ta
                        gaps.append(accumulated)
                    payback = next((i + 1 for i in range(len(gaps)) if all(v <= 0 for v in gaps[i:])), None)
                    pairs.append(
                        {
                            "online": a,
                            "cached": b,
                            "training_sustained_payback_step": payback,
                            "training_final_cached_minus_online_seconds": gaps[-1],
                            "scope": "sum of slowest-rank step wall times; startup/exit excluded",
                        }
                    )

                def throughput(mode):
                    return statistics.mean(r["training"]["samples_per_second"] for r in modes[mode])

                compared["full_training_throughput_improvement"] = throughput("cached") / throughput("online") - 1
                comparisons[name] = {"comparison": compared, "pairs": pairs}
            result["campaigns"][campaign["id"]] = {
                "source_commit": campaign["source_commit"],
                "scope": campaign["scope"],
                "runs": runs,
                "groups": comparisons,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional derived JSON output; never changes evidence")
    args = parser.parse_args()
    result = recompute()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for campaign, value in result["campaigns"].items():
        print(f"{campaign}: {value['source_commit']}; {value['scope']}")
        for name, group in value["groups"].items():
            c = group["comparison"]
            a, b = c["online_steady_samples_per_second"], c["cached_steady_samples_per_second"]
            print(
                f"  {name}: n={a['runs']}, {a['mean']:.6f} -> {b['mean']:.6f} samples/s, {c['steady_throughput_improvement']:+.2%}"
            )
        for name, run in value["runs"].items():
            if "numerical" in run:
                print(f"  {name}: stored report exact; {run['numerical']['tensor_counts']}; tensors not recomputed")


if __name__ == "__main__":
    main()
