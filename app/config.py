"""Application configuration.

All settings are read from environment variables (optionally via a `.env`
file) into a single Pydantic ``Settings`` object. Secrets are ONLY ever read
here — other modules must consume them through :func:`get_settings` and never
hardcode credentials.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (two levels above this file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SYSTEM_PROMPT = (
    "You are {agent_name}, a friendly and professional outbound calling agent "
    "for {company_name}. Speak naturally and concisely. Never invent facts; "
    "use the provided context only. Be polite, confirm any changes you make, "
    "and end the call warmly when the caller is done.\n\n"
    "Call context:\n{context_block}"
)


class Settings(BaseSettings):
    """Typed application settings populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ── Groq (LLM: chat completions for reasoning/summaries) ───────────────
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com"
    groq_chat_model: str = "llama-3.3-70b-versatile"

    # ── Embeddings (HuggingFace sentence-transformers, local — no API key) ──
    # all-MiniLM-L6-v2 produces 384-dimensional vectors.
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # ── Deepgram (speech: streaming STT + TTS) ──────────────────────────────
    deepgram_api_key: str = ""
    deepgram_stt_ws_url: str = "wss://api.deepgram.com/v1/listen"
    deepgram_stt_model: str = "nova-2"
    deepgram_tts_url: str = "https://api.deepgram.com/v1/speak"
    deepgram_tts_model: str = "aura-2-asteria-en"

    # ── Twilio ─────────────────────────────────────────────────────────────
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    twilio_webhook_base_url: str = "http://localhost:8000"

    # ── PostgreSQL / pgvector ──────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/calling_agent"
    # 384 = all-MiniLM-L6-v2 output dims. Must match the migration schema.
    pgvector_dimension: int = 384

    # ── Agent behaviour ────────────────────────────────────────────────────
    agent_name: str = "Alex"
    company_name: str = "Acme Corp"
    agent_system_prompt_file: str = "prompts/system.txt"
    memory_retrieval_top_k: int = 5
    max_turns_before_close: int = 20
    escalation_keywords: str = "manager,supervisor,complaint,cancel"
    escalation_webhook_url: str = ""

    # ── Barge-in / interruption ────────────────────────────────────────────
    barge_in_enabled: bool = True
    barge_in_min_speech_ms: int = 300          # ignore sounds shorter than this
    barge_in_trigger: str = "speech_started"    # speech_started | interim | both
    barge_in_filler_tokens: str = "uh-huh,uh huh,mm-hmm,mmhmm,yeah,yep,mm,uh,ah,hm,hmm"
    barge_in_vad_events: bool = True            # request Deepgram SpeechStarted events

    # ── Derived helpers ────────────────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def is_test(self) -> bool:
        return self.app_env.lower() == "test"

    @property
    def escalation_keyword_list(self) -> List[str]:
        """Escalation trigger keywords as a cleaned, lower-cased list."""
        return [k.strip().lower() for k in self.escalation_keywords.split(",") if k.strip()]

    @property
    def barge_in_filler_token_set(self) -> set:
        """Filler/backchannel tokens that must NOT trigger barge-in."""
        return {
            k.strip().lower()
            for k in self.barge_in_filler_tokens.split(",")
            if k.strip()
        }

    @property
    def deepgram_stt_listen_url(self) -> str:
        """Deepgram streaming-STT WebSocket URL tuned for Twilio 8 kHz μ-law."""
        from urllib.parse import urlencode

        # vad_events=true is required for Deepgram SpeechStarted events used
        # by barge-in detection. It is plan-dependent; disable via
        # BARGING_VAD_EVENTS=false if your Deepgram tier does not support it.
        params: Dict[str, str] = {
            "model": self.deepgram_stt_model,
            "encoding": "mulaw",
            "sample_rate": 8000,
            "channels": 1,
            "punctuate": "true",
            "interim_results": "true",  # required for speech_final + interim
            "endpointing": 300,  # ms of silence before an endpoint is declared
            "smart_format": "true",
        }
        if self.barge_in_vad_events:
            params["vad_events"] = "true"
        return f"{self.deepgram_stt_ws_url}?{urlencode(params)}"

    @property
    def deepgram_tts_speak_url(self) -> str:
        """Deepgram TTS URL returning raw 8 kHz μ-law audio (Twilio-ready)."""
        from urllib.parse import urlencode

        params = urlencode(
            {
                "model": self.deepgram_tts_model,
                "encoding": "mulaw",
                "sample_rate": 8000,
                "container": "none",
            }
        )
        return f"{self.deepgram_tts_url}?{params}"

    @property
    def asyncpg_dsn(self) -> str:
        """Plain ``postgresql://`` DSN for raw asyncpg connections."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")

    @property
    def webhook_base_ws(self) -> str:
        """The public webhook base URL converted to the ``wss`` scheme."""
        base = self.twilio_webhook_base_url.rstrip("/")
        if base.startswith("https://"):
            return "wss://" + base[len("https://"):]
        if base.startswith("http://"):
            return "ws://" + base[len("http://"):]
        return base

    def media_stream_ws_url(self, call_sid: str, context_json: str | None = None) -> str:
        """Fully qualified ``wss://`` URL Twilio should open a Media Stream to.

        ``context_json`` (a JSON-serialised call context) is forwarded as a
        query parameter so the WebSocket handler can reconstruct it.
        """
        url = f"{self.webhook_base_ws}/ws/media-stream/{call_sid}"
        if context_json:
            url += "?" + urlencode({"context": context_json})
        return url

    @property
    def _is_ngrok_url(self) -> bool:
        """True when the public webhook base URL is an ngrok tunnel."""
        return "ngrok" in self.twilio_webhook_base_url

    def twiml_callback_url(self, context: Dict[str, Any] | None = None) -> str:
        """HTTP(S) URL Twilio requests when the outbound call is answered.

        Embeds call context as a URL-safe query param so the media-stream
        WebSocket handler can reconstruct it and pass it to the agent.

        Also injects ``ngrok-skip-browser-warning=true`` for ngrok tunnels so
        that Twilio bypasses the browser interstitial page and receives the
        real TwiML XML response.
        """
        import json

        params: Dict[str, Any] = {}
        if self._is_ngrok_url:
            params["ngrok-skip-browser-warning"] = "true"
        if context:
            params["context"] = json.dumps(context)

        base = f"{self.twilio_webhook_base_url.rstrip('/')}/webhooks/twilio/twiml"
        if params:
            base += "?" + urlencode(params)
        return base

    def status_callback_url(self) -> str:
        """HTTP(S) URL Twilio posts lifecycle events to."""
        base = f"{self.twilio_webhook_base_url.rstrip('/')}/webhooks/twilio/status"
        if self._is_ngrok_url:
            base += "?" + urlencode({"ngrok-skip-browser-warning": "true"})
        return base

    def load_system_prompt(self, context: dict | None = None) -> str:
        """Load and render the agent system prompt template.

        Falls back to a built-in default if the configured prompt file is
        missing or unreadable. Template placeholders: ``{agent_name}``,
        ``{company_name}``, ``{context_block}``.
        """
        path = Path(self.agent_system_prompt_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            template = path.read_text(encoding="utf-8")
        except OSError:
            template = DEFAULT_SYSTEM_PROMPT

        context = context or {}
        context_block = "\n".join(f"- {key}: {value}" for key, value in context.items()) or "- (none)"
        try:
            return template.format(
                agent_name=self.agent_name,
                company_name=self.company_name,
                context_block=context_block,
            )
        except (KeyError, IndexError):
            # Template contains unknown placeholders — return it verbatim.
            return template


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()


__all__ = ["Settings", "get_settings", "PROJECT_ROOT", "DEFAULT_SYSTEM_PROMPT"]
