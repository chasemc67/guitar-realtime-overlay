"""Targeted vision-LLM calls through the Vercel AI Gateway (OpenAI-compatible API).

- base_url https://ai-gateway.vercel.sh/v1, key from the env var AI_GATEWAY_API_KEY (never printed, logged, cached
  or committed; `.env` is gitignored and loaded if present).
- Every response is cached on disk (`.cache/ai/`, gitignored), keyed by sha256(model, task, prompt, images), so
  re-runs are free and deterministic. Real calls are appended to `.cache/ai/usage.jsonl` with token counts and cost
  (prices from the gateway's /v1/models, cached).
- The pipeline never requires this module: `AIClient.available` is False without a key (or with --no-ai) and callers
  fall back to deterministic behaviour.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BASE_URL = "https://ai-gateway.vercel.sh/v1"
KEY_ENV = "AI_GATEWAY_API_KEY"
DEFAULT_MODEL = "anthropic/claude-opus-5.5"  # best Claude vision model on the gateway (2026-10)
FAST_MODEL = "anthropic/claude-sonnet-5.5"
ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / ".cache" / "ai"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip().strip("'\"")


def encode_image(img: np.ndarray, max_side: int = 1600, quality: int = 88) -> str:
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    if k < 1.0:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def parse_json(text: str) -> dict | None:
    """First JSON object in the text (models sometimes wrap it in prose or ``` fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    s = m.group(0)
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_task: dict = field(default_factory=dict)

    def add(self, task: str, rec: dict, cached: bool) -> None:
        t = self.by_task.setdefault(task, {"calls": 0, "cached": 0, "cost_usd": 0.0})
        if cached:
            self.cached += 1
            t["cached"] += 1
            return
        self.calls += 1
        t["calls"] += 1
        self.input_tokens += rec.get("input_tokens", 0)
        self.output_tokens += rec.get("output_tokens", 0)
        self.cost_usd += rec.get("cost_usd", 0.0)
        t["cost_usd"] += rec.get("cost_usd", 0.0)

    def summary(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class AIClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        enabled: bool = True,
        cache_dir: Path = CACHE_DIR,
        max_calls: int = 200,
        timeout: float = 120.0,
        offline: bool = False,
    ):
        """`offline`: serve cached responses only (never call the API)."""
        _load_dotenv(ROOT / ".env")
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.max_calls = max_calls
        self.timeout = timeout
        self.offline = offline
        self._key = os.environ.get(KEY_ENV, "") if enabled else ""
        self.enabled = enabled
        self.usage = Usage()
        self._prices: dict | None = None
        self.last_error = ""

    @property
    def available(self) -> bool:
        return self.enabled and (bool(self._key) or self.offline)

    # -- pricing -----------------------------------------------------------------------------------------------

    def prices(self, model: str) -> tuple[float, float]:
        """(input, output) USD per token from the gateway model list (cached for a week)."""
        if self._prices is None:
            p = self.cache_dir / "models.json"
            data = None
            if p.exists() and time.time() - p.stat().st_mtime < 7 * 86400:
                data = json.loads(p.read_text())
            elif self._key and not self.offline:
                try:
                    req = urllib.request.Request(BASE_URL + "/models", headers={"Authorization": f"Bearer {self._key}"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        data = json.loads(r.read())
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(data))
                except Exception:  # noqa: BLE001 - pricing is best effort
                    data = None
            self._prices = {}
            for m in (data or {}).get("data", []):
                pr = m.get("pricing") or {}
                try:
                    self._prices[m["id"]] = (float(pr.get("input", 0)), float(pr.get("output", 0)))
                except (TypeError, ValueError):
                    pass
        return self._prices.get(model, (0.0, 0.0))

    # -- calls -------------------------------------------------------------------------------------------------

    def _key_for(self, model, task, prompt, images_b64, max_tokens) -> str:
        h = hashlib.sha256()
        for part in (model, task, prompt, str(max_tokens), *images_b64):
            h.update(part.encode())
            h.update(b"\0")
        return h.hexdigest()

    def chat_json(
        self,
        task: str,
        prompt: str,
        images: list[np.ndarray],
        model: str | None = None,
        max_tokens: int = 4000,
        system: str | None = None,
    ) -> dict | None:
        """Ask a vision model; returns the parsed JSON object (or None). Cached by content."""
        model = model or self.model
        imgs = [encode_image(im) for im in images]
        full_prompt = (system or "") + "\n\n" + prompt
        key = self._key_for(model, task, full_prompt, imgs, max_tokens)
        path = self.cache_dir / task / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            self.usage.add(task, rec, cached=True)
            return rec.get("parsed")
        if not self.available or self.offline:
            return None
        if self.usage.calls >= self.max_calls:
            self.last_error = "AI call budget exhausted"
            return None
        content = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        t0 = time.time()
        try:
            text, usage = self._post(model, messages, max_tokens)
        except Exception as e:  # noqa: BLE001 - network/API errors degrade to "no answer"
            self.usage.failed += 1
            self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            return None
        pin, pout = self.prices(model)
        rec = {
            "task": task,
            "model": model,
            "prompt": prompt,
            "system": system,
            "n_images": len(imgs),
            "response": text,
            "parsed": parse_json(text),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cost_usd": usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout,
            "seconds": round(time.time() - t0, 2),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        if rec["parsed"] is not None:  # unparsable answers (e.g. reasoning ate the budget) are retried next time
            path.write_text(json.dumps(rec, indent=1))
        else:
            self.last_error = f"unparsable answer ({rec['output_tokens']} output tokens)"
        with (self.cache_dir / "usage.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        k: rec[k]
                        for k in ("time", "task", "model", "input_tokens", "output_tokens", "cost_usd", "seconds")
                    }
                )
                + "\n"
            )
        self.usage.add(task, rec, cached=False)
        return rec["parsed"]

    def _post(self, model: str, messages: list, max_tokens: int) -> tuple[str, dict]:
        try:
            from openai import OpenAI  # optional dependency (`uv sync --extra ai`)
        except ImportError:
            OpenAI = None  # noqa: N806
        if OpenAI is not None:
            client = OpenAI(api_key=self._key, base_url=BASE_URL, timeout=self.timeout, max_retries=2)
            r = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0)
            u = r.usage
            return r.choices[0].message.content or "", {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            }
        body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}).encode()
        req = urllib.request.Request(
            BASE_URL + "/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"]["content"] or "", d.get("usage", {})
