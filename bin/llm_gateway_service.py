import argparse
import logging
import sys

from kn.config import load_configs
from kn.llm_gateway.runner import run_gateway_service


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the LLM gateway service loop")
    parser.add_argument("--service", default=None, help="Service name (defaults to llm.service from config)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...)")
    parser.add_argument("--once", action="store_true", help="Process a single request and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    cfg = load_configs()
    service = args.service or cfg.get("llm", {}).get("service", "lmstudio")
    run_gateway_service(service_name=service, loop_forever=not args.once, cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
