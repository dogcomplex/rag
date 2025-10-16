import argparse
import threading
import time
import sys

from kn.config import load_configs
from kn.llm_gateway.runner import run_gateway_service
from kn.llm_gateway.client import submit_chat_request
from kn.llm_gateway.errors import GatewayError, QueueTimeoutError, RequestRejectedError


def _run_gateway(service: str, cfg):
    run_gateway_service(service, loop_forever=True, cfg=cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Smoke-test the LLM gateway by issuing a simple chat request.")
    parser.add_argument("prompt", nargs="?", default="Reply with OK.", help="Prompt to send through the gateway")
    parser.add_argument("--service", default=None, help="Service name to target (defaults to llm.service from config)")
    parser.add_argument("--model", default=None, help="Override model name for the request")
    parser.add_argument("--boot-delay", type=float, default=0.5, help="Seconds to wait after starting the gateway thread")
    args = parser.parse_args(argv)

    cfg = load_configs()
    service = args.service or cfg.get("llm", {}).get("service", "lmstudio")

    gateway_thread = threading.Thread(target=_run_gateway, args=(service, cfg), daemon=True)
    gateway_thread.start()
    time.sleep(max(args.boot_delay, 0))

    overrides = {"model": args.model} if args.model else None
    try:
        response = submit_chat_request(
            args.prompt,
            overrides=overrides,
            cfg=cfg,
        )
    except (QueueTimeoutError, RequestRejectedError, GatewayError) as err:
        print(f"Gateway call failed: {err}")
        return 1
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())






