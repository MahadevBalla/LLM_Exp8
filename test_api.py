import csv
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

# Usage: python test_api.py [base_url]   (default: http://127.0.0.1:8000)
BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
RESULTS_FILE = Path("test_results.csv")


def run_case(name, method, path, payload=None):
    started = time.perf_counter()
    try:
        if method == "GET":
            response = requests.get(BASE_URL + path, timeout=180)
        else:
            response = requests.post(BASE_URL + path, json=payload, timeout=180)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text[:200]}
        return {
            "test_case": name,
            "http_status": response.status_code,
            "client_latency_ms": elapsed_ms,
            "server_latency_ms": body.get("response_time_ms", ""),
            "passed": (
                (name == "Health endpoint" and response.status_code == 200)
                or (name == "Frontend page" and response.status_code == 200)
                or (name == "Supported question" and response.status_code == 200 and bool(body.get("answer")))
                or (name == "Unsupported question" and response.status_code == 200 and bool(body.get("answer")))
                or (name == "Empty question" and response.status_code == 422)
                or (name == "Overlong question" and response.status_code == 422)
            ),
            "detail": str(body.get("detail", body.get("status", "")))[:180],
        }
    except requests.RequestException as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "test_case": name,
            "http_status": "NO_RESPONSE",
            "client_latency_ms": elapsed_ms,
            "server_latency_ms": "",
            "passed": False,
            "detail": type(exc).__name__,
        }


def main():
    cases = [
        ("Health endpoint", "GET", "/health", None),
        ("Frontend page", "GET", "/", None),
        ("Supported question", "POST", "/ask", {
            "question": "What should someone do about a missed medicine dose?"
        }),
        ("Unsupported question", "POST", "/ask", {
            "question": "What are the recommended treatments for chronic migraine?"
        }),
        ("Empty question", "POST", "/ask", {"question": "   "}),
        ("Overlong question", "POST", "/ask", {"question": "x" * 1001}),
    ]

    results = [run_case(*case) for case in cases]

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nTesting {BASE_URL}")
    print("Experiment 7 API test results")
    print("-" * 78)
    for row in results:
        print(
            f"{row['test_case']:<24} status={row['http_status']!s:<12} "
            f"latency={row['client_latency_ms']} ms  pass={row['passed']}"
        )
    print(f"\nSaved detailed results to: {RESULTS_FILE.resolve()}")

    # Plot 1: actual client-side latency measured for each endpoint test.
    latency_rows = [r for r in results if isinstance(r["http_status"], int)]
    if latency_rows:
        labels = [r["test_case"] for r in latency_rows]
        values = [r["client_latency_ms"] for r in latency_rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, values, edgecolor="black")
        ax.set_title("Experiment 7: API Response Time by Test Case")
        ax.set_xlabel("Test case")
        ax.set_ylabel("Client-side latency (milliseconds)")
        ax.tick_params(axis="x", labelrotation=20)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.0f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig("plot_response_times.png", dpi=160)

    # Plot 2: HTTP status distribution from the actual test run.
    status_counts = {}
    for row in results:
        label = str(row["http_status"])
        status_counts[label] = status_counts.get(label, 0) + 1

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(status_counts.keys()), list(status_counts.values()), edgecolor="black")
    ax.set_title("Experiment 7: HTTP Status Codes Observed")
    ax.set_xlabel("HTTP status")
    ax.set_ylabel("Number of test cases")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(bar.get_height())}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig("plot_http_status.png", dpi=160)


if __name__ == "__main__":
    main()
