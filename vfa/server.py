"""Local HTTP server for the browser voice UI.

Single-threaded on purpose (one caller at a time) so the orchestrator and its
SQLite shadow log stay on one thread. Builds the pipeline ONCE at startup. The server does
the real STT (Deepgram) and TTS (ElevenLabs) itself; the orchestrator handles the logic on
text and its internal TTS stays a no-op fake.

Run:  VFA_REAL_MODELS=1 ./.venv312/bin/python -m vfa.server   (then open http://localhost:8000)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from . import smalltalk
from .config import load_settings
from .factory import build_auto
from .kb.ingest import load_kb
from .prompts import GREETING

WEB = Path(__file__).parent / "web"
_VARIANTS = {"lexical", "semantic", "auto", "lexical_only"}
FILLERS = ["Sure, let me check that for you.", "One moment while I look that up.",
           "Let me find that for you.", "Okay, just a second.", "Good question — let me see.",
           "Let me pull that up for you.", "Alright, checking on that now.",
           "Give me just a moment."]


class Agent:
    """Holds the one built pipeline + the cloud STT/TTS, guarded by a lock."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.chunks = load_kb()
        self.orch, self.backends = build_auto(self.settings, self.chunks)
        self.variant = self.settings.ab_variant
        self.call_id = f"web-{int(time.time())}"
        self.turn = 0
        self.lock = threading.Lock()
        self.deepgram = self._deepgram()
        self.eleven = self._eleven()
        self._fillers: dict[str, str] = {}   # phrase -> audio_b64 (pre-synth once at startup)
        for _ph in FILLERS:
            self._fillers[_ph] = self.tts_b64(_ph)
        # Pre-synthesize the greeting + every canned small-talk reply once, so those turns are
        # INSTANT and spend no TTS credits per call. Small talk then never waits on live
        # synthesis (which is what used to push a simple "hi" past the client filler timer).
        self._canned: dict[str, str] = {GREETING: self.tts_b64(GREETING)}
        for _ph in smalltalk.REPLIES:
            self._canned[_ph] = self.tts_b64(_ph)

    def _deepgram(self):
        if self.settings.deepgram_api_key:
            from .stt.deepgram import DeepgramTranscriber
            return DeepgramTranscriber(self.settings.deepgram_api_key)
        return None

    def _eleven(self):
        s = self.settings
        if s.elevenlabs_api_key and s.elevenlabs_voice_id:
            from .tts.elevenlabs import ElevenLabsTTS
            return ElevenLabsTTS(s.elevenlabs_api_key, s.elevenlabs_voice_id, s.elevenlabs_model_id)
        return None

    def set_variant(self, v: str) -> None:
        if v in _VARIANTS:
            self.variant = v

    def filler(self) -> dict:
        import random
        text = random.choice(FILLERS)
        return {"text": text, "audio_b64": self._fillers.get(text, "")}

    def all_fillers(self) -> list[str]:
        return [self._fillers[p] for p in FILLERS if self._fillers.get(p)]

    def report(self) -> dict:
        """Unanswered-questions report built live from this run's shadow log."""
        from dataclasses import asdict
        from .report import build_report, load_turns
        try:
            return asdict(build_report(load_turns(self.settings.shadow_db)))
        except Exception as e:
            return {"error": str(e)}

    def tts_b64(self, text: str) -> str:
        if not (text and self.eleven):
            return ""
        # Persistent disk cache: synthesize each distinct phrase at most once EVER (keyed by
        # voice+model+text), so greetings / fillers / small-talk audio are NOT re-synthesized
        # on every restart — that avoids re-spending ElevenLabs credits. (data/ is gitignored)
        s = self.settings
        cdir = os.path.join(s.audio_dir, "tts_cache")
        key = hashlib.sha256(f"{s.elevenlabs_voice_id}|{s.elevenlabs_model_id}|{text}".encode()).hexdigest()[:20]
        path = os.path.join(cdir, key + ".mp3")
        try:
            if os.path.exists(path):
                return base64.b64encode(open(path, "rb").read()).decode("ascii")
        except Exception:
            pass
        try:
            data = self.eleven.synthesize_bytes(text)
        except Exception:
            return ""   # graceful (e.g. quota exceeded): UI still shows the text
        try:
            os.makedirs(cdir, exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
        except Exception:
            pass
        return base64.b64encode(data).decode("ascii")

    def turn_from_text(self, text: str, stt_ms: float = 0.0) -> dict:
        with self.lock:
            self.orch.ab.variant = self.variant
            self.turn += 1
            t0 = time.perf_counter()
            r = self.orch.handle_turn(self.call_id, self.turn, text)
            pipeline_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        # small-talk / greeting replies are pre-synth'd -> served from cache (instant, free);
        # only real answers hit live TTS.
        audio = self._canned.get(r.response or "") or self.tts_b64(r.response or "")
        tts_ms = (time.perf_counter() - t1) * 1000.0
        return {
            "transcript": r.utterance,
            "response": r.response,
            "outcome": r.outcome,
            "served_path": r.served_path,
            "latency_ms": round(r.total_latency_ms),
            "stt_ms": round(stt_ms), "pipeline_ms": round(pipeline_ms), "tts_ms": round(tts_ms),
            "variant": self.variant,
            "gate": r.gate.reason,
            "lexical_candidate": r.lexical.candidate_response,
            "semantic_candidate": (r.semantic.candidate_response if r.semantic else None),
            "audio_b64": audio,
        }


AGENT: Agent | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name, ctype):
        p = WEB / name
        if not p.exists():
            self._json({"error": "not found"}, 404)
            return
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._file("index.html", "text/html; charset=utf-8")
        elif self.path == "/app.js":
            self._file("app.js", "application/javascript")
        elif self.path == "/style.css":
            self._file("style.css", "text/css")
        elif self.path == "/api/health":
            self._json({"ok": True, "variant": AGENT.variant, "backends": AGENT.backends,
                        "stt": bool(AGENT.deepgram), "tts": bool(AGENT.eleven)})
        elif self.path == "/api/variant":
            self._json({"variant": AGENT.variant})
        elif self.path == "/api/greeting":
            self._json({"response": GREETING, "audio_b64": AGENT._canned.get(GREETING, "")})
        elif self.path == "/api/filler":
            self._json(AGENT.filler())
        elif self.path == "/api/fillers":
            self._json({"fillers": AGENT.all_fillers()})
        elif self.path == "/api/report":
            self._json(AGENT.report())
        else:
            self._json({"error": "not found"}, 404)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def do_POST(self):
        if self.path == "/api/variant":
            data = json.loads(self._body() or b"{}")
            AGENT.set_variant(data.get("variant", ""))
            self._json({"variant": AGENT.variant})
        elif self.path == "/api/turn_text":
            data = json.loads(self._body() or b"{}")
            if data.get("variant"):
                AGENT.set_variant(data["variant"])
            text = (data.get("text") or "").strip()
            self._json(AGENT.turn_from_text(text) if text else {"transcript": "", "response": None,
                                                                "outcome": "no_input", "audio_b64": ""})
        elif self.path == "/api/turn_audio":
            if not AGENT.deepgram:
                self._json({"error": "no STT configured"}, 400)
                return
            v = self.headers.get("X-Variant")
            if v:
                AGENT.set_variant(v)
            ctype = self.headers.get("Content-Type", "audio/webm")
            t0 = time.perf_counter()
            try:
                text = AGENT.deepgram.transcribe_bytes(self._body(), ctype).strip()
            except Exception as e:
                self._json({"error": f"stt failed: {e}"}, 502)
                return
            stt_ms = (time.perf_counter() - t0) * 1000.0
            self._json(AGENT.turn_from_text(text, stt_ms) if text else {"transcript": "", "response": None,
                                                                        "outcome": "no_input", "audio_b64": ""})
        else:
            self._json({"error": "not found"}, 404)


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    global AGENT
    print("Building pipeline (loading models, indexing KB)…")
    AGENT = Agent()
    print(f"Ready. Backends: {AGENT.backends}")
    print(f"STT={'Deepgram' if AGENT.deepgram else 'OFF'}  TTS={'ElevenLabs' if AGENT.eleven else 'OFF'}")
    print(f"Open  http://{host}:{port}")
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
