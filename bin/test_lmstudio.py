import argparse
import os
import sys
import textwrap
from typing import Iterable

import requests

try:
    from kn.config import load_configs
except Exception:  # noqa: BLE001
    load_configs = None  # type: ignore

DEFAULT_PROMPT = "This is a connectivity test. Reply with a short acknowledgement."


def _print_header(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def list_models(base_url: str, timeout: int = 5) -> list[str]:
    url = base_url.rstrip('/') + '/models'
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    models = [m.get('id') for m in data.get('data', []) if m.get('id')]
    return models


def ping_model(base_url: str, model: str, prompt: str, timeout: int = 30, max_tokens: int = 32,
               temperature: float | None = None) -> tuple[int, str | None]:
    url = base_url.rstrip('/') + '/chat/completions'
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
    }
    if temperature is not None:
        payload['temperature'] = temperature
    resp = requests.post(url, json=payload, timeout=timeout)
    status = resp.status_code
    content = None
    if resp.ok:
        try:
            content = resp.json()['choices'][0]['message']['content']
        except Exception as exc:  # noqa: BLE001
            content = f"Failed to parse response: {exc}"  # type: ignore[assignment]
    else:
        try:
            content = resp.text
        except Exception:  # noqa: BLE001
            content = None
    return status, content


def check_dashboard(base_url: str, timeout: int = 5) -> dict[str, tuple[int, str]]:
    """Ping key dashboard endpoints and return {endpoint: (status, details)}."""
    results: dict[str, tuple[int, str]] = {}
    base = base_url.rstrip('/')

    def _record(name: str, response: requests.Response, body_preview: str | None = None) -> None:
        snippet = ""
        if body_preview is not None:
            snippet = body_preview
        else:
            try:
                snippet = textwrap.shorten(response.text or '', width=120, placeholder='…')
            except Exception:  # noqa: BLE001
                snippet = "<no body>"
        results[name] = (response.status_code, snippet)

    try:
        resp = requests.get(f"{base}/api/status", timeout=timeout)
        if resp.ok:
            body_preview = textwrap.shorten(resp.text, width=120, placeholder='…')
        else:
            body_preview = resp.text
        _record('GET /api/status', resp, body_preview)
    except Exception as exc:  # noqa: BLE001
        results['GET /api/status'] = (0, f"{exc}")

    try:
        resp = requests.get(f"{base}/api/queue/list?limit=5", timeout=timeout)
        _record('GET /api/queue/list', resp)
    except Exception as exc:  # noqa: BLE001
        results['GET /api/queue/list'] = (0, f"{exc}")

    try:
        payload = {'mode': 'reset-running'}
        resp = requests.post(f"{base}/api/queue/clear", json=payload, timeout=timeout)
        _record('POST /api/queue/clear', resp)
    except Exception as exc:  # noqa: BLE001
        results['POST /api/queue/clear'] = (0, f"{exc}")

    try:
        resp = requests.get(f"{base}/api/workers", timeout=timeout)
        _record('GET /api/workers', resp)
    except Exception as exc:  # noqa: BLE001
        results['GET /api/workers'] = (0, f"{exc}")

    return results


def summarize_results(title: str, items: Iterable[tuple[str, tuple[int, str]]]) -> None:
    _print_header(title)
    for name, (status, detail) in items:
        prefix = 'OK' if status == 200 else 'ERR'
        print(f"[{prefix}] {name} -> status={status} detail={detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check LM Studio availability and dashboard endpoints.")
    parser.add_argument('--base-url', default=os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:12345/v1'),
                        help='LM Studio base URL (default: %(default)s or OPENAI_BASE_URL env)')
    parser.add_argument('--dashboard-url', default='http://127.0.0.1:5051',
                        help='Dashboard base URL (default: %(default)s)')
    parser.add_argument('--models', nargs='*', default=None,
                        help='Specific model IDs to probe. If absent, lists models only.')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='Prompt text for chat check.')
    parser.add_argument('--max-tokens', type=int, default=32, help='max_tokens for chat check (default: %(default)s)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Optional temperature value for chat checks.')
    parser.add_argument('--timeout', type=int, default=30, help='HTTP timeout in seconds (default: %(default)s)')
    parser.add_argument('--skip-dashboard', action='store_true', help='Skip dashboard API checks.')
    parser.add_argument('--plugins', nargs='*', default=None,
                        help='Plugin names (from config) to resolve into model IDs for chat checks.')
    parser.add_argument('--show-response', action='store_true', help='Print the full model response instead of a preview.')
    args = parser.parse_args(argv)

    base = args.base_url.rstrip('/')
    _print_header(f"LM Studio Diagnostics @ {base}")

    try:
        models = list_models(base, timeout=args.timeout)
    except requests.RequestException as exc:
        print(f"[ERROR] Could not reach {base + '/models'}: {exc}")
        return 1

    if not models:
        print("[WARN] No models returned by the server.")
    else:
        print("Available models:")
        for model in models:
            print(f"  - {model}")

    plugin_models: list[str] = []
    if args.plugins:
        if load_configs is None:
            print("[WARN] kn.config.load_configs not available; skipping plugin resolution.")
        else:
            cfg = load_configs()
            for plugin in args.plugins:
                p = cfg.get('plugins', {}).get(plugin)
                if not isinstance(p, dict):
                    print(f"[WARN] Plugin '{plugin}' not found in config; skipping")
                    continue
                llm_cfg = p.get('llm') if isinstance(p.get('llm'), dict) else {}
                model_id = llm_cfg.get('model') or cfg.get('llm', {}).get('model')
                if not model_id:
                    print(f"[WARN] Plugin '{plugin}' has no model configured; skipping")
                    continue
                plugin_models.append(model_id)
                print(f"Resolved plugin '{plugin}' -> model '{model_id}'")

    models_to_check = []
    seen = set()
    for model in [*(args.models or []), *plugin_models]:
        m = model.strip()
        if not m or m in seen:
            continue
        seen.add(m)
        models_to_check.append(m)

    if models_to_check:
        _print_header("Chat Checks")
        for model in models_to_check:
            model = model.strip()
            if not model:
                continue
            print(f"Testing model: {model}")
            try:
                status, content = ping_model(base, model, prompt=args.prompt,
                                             timeout=args.timeout, max_tokens=args.max_tokens,
                                             temperature=args.temperature)
            except requests.RequestException as exc:
                print(f"  [ERROR] Request failed: {exc}")
                continue

            if status == 200:
                if args.show_response:
                    print(f"  [OK] status=200 response={content!r}")
                else:
                    preview = textwrap.shorten(content or '', width=120, placeholder='…')
                    print(f"  [OK] status=200 response={preview!r} (use --show-response to display full text)")
            else:
                snippet = textwrap.shorten(content or '', width=120, placeholder='…')
                print(f"  [FAIL] status={status} body={snippet!r}")
    else:
        print("\nNo specific models requested for chat check. Use --models or --plugins to verify.")

    if not args.skip_dashboard:
        dashboard = args.dashboard_url.rstrip('/')
        results = check_dashboard(dashboard, timeout=args.timeout)
        summarize_results(f"Dashboard Diagnostics @ {dashboard}", sorted(results.items()))

    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
