import argparse
import sys

from kn.config import load_configs
from kn.jobs_sqlite import reset_running_jobs, reset_counter, get_counter


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reset job status and counters for the enrichment queue")
    parser.add_argument("--status", default="pending", help="Status to set for running jobs (default: pending)")
    parser.add_argument("--counter", default="llm_concurrency", help="Counter name to reset (default: llm_concurrency)")
    parser.add_argument("--no-counter", action="store_true", help="Skip resetting the counter")
    args = parser.parse_args(argv)

    cfg = load_configs()

    ids = reset_running_jobs(cfg, status=args.status)
    print(f"Reset {len(ids)} jobs from 'running' to '{args.status}'")

    if not args.no_counter:
        before = get_counter(cfg, args.counter)
        reset_counter(cfg, args.counter)
        after = get_counter(cfg, args.counter)
        print(f"Counter '{args.counter}' reset (was {before}, now {after})")


if __name__ == "__main__":
    sys.exit(main())
