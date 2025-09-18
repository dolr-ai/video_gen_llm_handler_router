import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import defaultdict, deque

import httpx
from datetime import datetime


DEFAULT_URL = "https://yral-video-gen-llm-handler.fly.dev/v1/split"
DEFAULT_PROMPT = "A cinematic sunset over mountains with orchestral score."


class Metrics:
    def __init__(self, latency_sample_size: int = 100_000) -> None:
        self.started = 0
        self.done = 0
        self.ok = 0
        self.errors = 0
        self.status_counts: dict[int, int] = defaultdict(int)
        self.latencies_ms = deque(maxlen=latency_sample_size)

    def record(self, status_code: int | None, latency_ms: float | None) -> None:
        self.done += 1
        if status_code is None:
            self.errors += 1
            return
        self.status_counts[status_code] += 1
        if 200 <= status_code < 300:
            self.ok += 1
        else:
            self.errors += 1
        if latency_ms is not None:
            self.latencies_ms.append(latency_ms)

    def summarize(self) -> dict:
        lat_sorted = sorted(self.latencies_ms)
        def pct(p: float) -> float | None:
            if not lat_sorted:
                return None
            idx = min(len(lat_sorted) - 1, max(0, int(p * (len(lat_sorted) - 1))))
            return lat_sorted[idx]

        return {
            "started": self.started,
            "done": self.done,
            "ok": self.ok,
            "errors": self.errors,
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "status_counts": dict(self.status_counts),
        }


async def do_request(client: httpx.AsyncClient, url: str, payload: dict, headers: dict, metrics: Metrics) -> None:
    t0 = time.perf_counter()
    status = None
    try:
        resp = await client.post(url, json=payload, headers=headers)
        status = resp.status_code
    except Exception:
        status = None
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        metrics.record(status, latency_ms)


async def worker(worker_id: int, token_q: asyncio.Queue, client: httpx.AsyncClient, url: str, payload: dict, headers: dict, metrics: Metrics) -> None:
    while True:
        token = await token_q.get()
        if token is None:
            token_q.task_done()
            break
        await do_request(client, url, payload, headers, metrics)
        token_q.task_done()


async def generate_tokens(token_q: asyncio.Queue, rps: int, duration_s: float) -> None:
    # Generate tokens in small time slices to smooth out scheduling
    slice_s = 0.01  # 10ms slices
    tokens_per_slice = rps * slice_s
    carry = 0.0
    end_time = time.perf_counter() + duration_s
    while True:
        now = time.perf_counter()
        if now >= end_time:
            break
        carry += tokens_per_slice
        emit = int(carry)
        if emit > 0:
            carry -= emit
            for _ in range(emit):
                await token_q.put(1)
        await asyncio.sleep(slice_s)


async def run_load(url: str, prompt: str, rps: int, duration_s: float, concurrency: int, timeout_s: float, insecure: bool, report_every_s: float, log_file=None) -> dict:
    metrics = Metrics()
    headers = {"content-type": "application/json"}
    payload = {"prompt": prompt}

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(http2=True, timeout=timeout_s, verify=(not insecure), limits=limits) as client:
        token_q: asyncio.Queue = asyncio.Queue(maxsize=rps * 2 if rps > 0 else 1000)

        workers = max(1, concurrency)
        worker_tasks = [
            asyncio.create_task(worker(i, token_q, client, url, payload, headers, metrics))
            for i in range(workers)
        ]

        start = time.perf_counter()
        last_report = start
        last_done = 0

        gen_task = asyncio.create_task(generate_tokens(token_q, rps, duration_s))

        # Reporter loop
        async def reporter() -> None:
            nonlocal last_report, last_done
            while not gen_task.done() or (metrics.done < rps * duration_s):
                await asyncio.sleep(report_every_s)
                now = time.perf_counter()
                interval = now - last_report
                completed = metrics.done
                interval_done = completed - last_done
                inst_rps = interval_done / interval if interval > 0 else 0.0
                evt = {
                    "event": "progress",
                    "elapsed_s": round(now - start, 2),
                    "inst_rps": round(inst_rps, 2),
                    "done": completed,
                    "ok": metrics.ok,
                    "errors": metrics.errors,
                }
                line = json.dumps(evt)
                print(line)
                sys.stdout.flush()
                if log_file is not None:
                    try:
                        log_file.write(line + "\n")
                        log_file.flush()
                    except Exception:
                        pass
                last_report = now
                last_done = completed

        reporter_task = asyncio.create_task(reporter())

        await gen_task
        # Send stop signals for workers
        for _ in range(len(worker_tasks)):
            await token_q.put(None)

        await token_q.join()

        # Cancel reporter after queue drained
        reporter_task.cancel()
        try:
            await reporter_task
        except asyncio.CancelledError:
            pass

        for t in worker_tasks:
            await t

    total_time = time.perf_counter() - start
    summary = metrics.summarize()
    summary.update({
        "target_rps": rps,
        "achieved_rps": round(summary["done"] / total_time, 2) if total_time > 0 else 0.0,
        "duration_s": round(total_time, 2),
        "concurrency": concurrency,
        "url": url,
    })
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async stress test for the Fly.io prompt splitter API")
    p.add_argument("--url", default=DEFAULT_URL, help="Target URL for POST /v1/split")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text to send in request body")
    p.add_argument("--rps", type=int, default=10_000, help="Target requests per second")
    p.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds")
    p.add_argument("--concurrency", type=int, default=2_000, help="Max in-flight requests")
    p.add_argument("--timeout-s", type=float, default=10.0, help="Per-request timeout seconds")
    p.add_argument("--insecure", action="store_true", help="Disable TLS verification")
    p.add_argument("--report-every", type=float, default=1.0, help="Progress report interval seconds")
    p.add_argument("--out", default="stress_results", help="Write JSONL logs to this path. If directory or missing extension, creates run-<ts>.jsonl inside")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.rps <= 0 or args.duration <= 0:
        print("rps and duration must be > 0", file=sys.stderr)
        sys.exit(1)

    # Prepare logging file if provided
    log_path = None
    log_file = None
    path = args.out
    if path:
        # If path exists and is a directory: write run-<ts>.jsonl inside it
        # If path does not exist and has no .json/.jsonl extension: treat as directory
        # Else: treat as explicit file path
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        is_dir = os.path.isdir(path) or (not os.path.exists(path) and not (path.endswith(".json") or path.endswith(".jsonl")))
        if is_dir:
            os.makedirs(path, exist_ok=True)
            log_path = os.path.join(path, f"run-{ts}.jsonl")
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            log_path = path
        log_file = open(log_path, "a", buffering=1)

    start_evt = {
        "event": "start",
        "url": args.url,
        "rps": args.rps,
        "duration_s": args.duration,
        "concurrency": args.concurrency,
        "timeout_s": args.timeout_s,
        "prompt_len": len(args.prompt),
        "log_file": log_path,
    }
    line = json.dumps(start_evt)
    print(line)
    sys.stdout.flush()
    if log_file is not None:
        try:
            log_file.write(line + "\n")
            log_file.flush()
        except Exception:
            pass
    sys.stdout.flush()

    summary = asyncio.run(run_load(
        url=args.url,
        prompt=args.prompt,
        rps=args.rps,
        duration_s=args.duration,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
        insecure=args.insecure,
        report_every_s=args.report_every,
        log_file=log_file,
    ))
    done_evt = {"event": "done", **summary}
    line = json.dumps(done_evt, default=lambda o: o)
    print(line)
    if log_file is not None:
        try:
            log_file.write(line + "\n")
            log_file.flush()
            log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


