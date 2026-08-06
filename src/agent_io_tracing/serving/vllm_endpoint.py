"""Inspect an already-running OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _base_url(value: str) -> str:
    url = value.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


@dataclass(frozen=True)
class VLLMEndpoint:
    """Read and explicitly reset a remotely managed vLLM server."""

    base_url: str
    api_key: str = ""
    timeout_s: float = 10.0

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        accept_json: bool = True,
    ) -> Any:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{_base_url(self.base_url)}{path}",
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8", "replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"{method} {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, OSError) as exc:
            reason = getattr(exc, "reason", None) or str(exc)
            raise RuntimeError(
                f"{method} {path} could not reach {self.base_url}: {reason}"
            ) from exc
        if not accept_json:
            return body
        return json.loads(body) if body.strip() else {}

    def health(self) -> None:
        self._request("/health", accept_json=False)

    def models(self) -> list[str]:
        payload = self._request("/v1/models")
        return [
            str(item["id"])
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]

    def metrics(self) -> str:
        return str(self._request("/metrics", accept_json=False))

    def reset_prefix_cache(self) -> Any:
        return self._request("/reset_prefix_cache", method="POST")

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{_base_url(self.base_url)}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8", "replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"POST {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, OSError) as exc:
            reason = getattr(exc, "reason", None) or str(exc)
            raise RuntimeError(
                f"POST {path} could not reach {self.base_url}: {reason}"
            ) from exc
        return json.loads(body) if body.strip() else {}

    def tokenize_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **template_options: Any,
    ) -> list[int]:
        payload = {
            "model": model,
            "messages": messages,
            "return_token_strs": False,
            **template_options,
        }
        response = self.post_json("/tokenize", payload)
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) for token in tokens
        ):
            raise RuntimeError("/tokenize did not return integer token IDs")
        return tokens


def _metric_names(text: str) -> list[str]:
    wanted = (
        "prefix_cache",
        "cache_usage",
        "preempt",
        "time_to_first_token",
        "num_requests",
    )
    names = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if any(fragment in name for fragment in wanted):
            names.add(name)
    return sorted(names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a remotely managed vLLM endpoint.",
    )
    parser.add_argument(
        "--url",
        default=(
            os.environ.get("VLLM_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        ),
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("VLLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY_1")
            or ""
        ),
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the prefix cache after the read-only checks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.url:
        raise SystemExit(
            "Set VLLM_URL or OPENAI_BASE_URL, or pass --url."
        )
    endpoint = VLLMEndpoint(args.url, args.api_key, args.timeout)
    try:
        endpoint.health()
        models = endpoint.models()
        metrics = _metric_names(endpoint.metrics())
        reset_result = endpoint.reset_prefix_cache() if args.reset else None
    except (RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"vLLM endpoint check failed: {exc}") from exc
    print(json.dumps({
        "base_url": _base_url(args.url),
        "health": "ok",
        "models": models,
        "kv_metric_names": metrics,
        "prefix_cache_reset": reset_result if args.reset else "not requested",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
