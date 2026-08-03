"""Dograh-native NVIDIA PersonaPlex / moshi.server realtime adapter.

PersonaPlex is full-duplex speech-to-speech over a WebSocket binary protocol
(kind bytes: 0x00 handshake, 0x01 Opus, 0x02 text, 0x03 PCM float32). Dograh
owns this client (no upstream Pipecat service yet).

MVP contract:
- Deferred connect until the engine has a system instruction / greeting
- Optional gateway bearer via ``api_key`` (never HF_TOKEN)
- Local VAD path (Ultravox-like) — provider emits no turn frames
- Instruction changes reconnect after bot speech (Gemini Live pattern)
- No native tool calling
"""

from __future__ import annotations

import array
import asyncio
import struct
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

try:
    import audioop  # CPython < 3.13
except ModuleNotFoundError:  # pragma: no cover
    audioop = None  # type: ignore

from loguru import logger
from pydantic import Field
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMServiceMetadataFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.time import time_now_iso8601

# Moshi / PersonaPlex binary frame kinds
_KIND_HANDSHAKE = 0x00
_KIND_OPUS = 0x01
_KIND_TEXT = 0x02
_KIND_PCM_F32 = 0x03

_PERSONAPLEX_SAMPLE_RATE = 24000
_DEFAULT_VOICE = "NATF2"


def normalize_personaplex_voice(voice: str | None) -> str:
    """Return a moshi voice_prompt value (e.g. NATF2.pt)."""
    raw = (voice or _DEFAULT_VOICE).strip()
    if not raw:
        raw = _DEFAULT_VOICE
    if not raw.endswith(".pt"):
        raw = f"{raw}.pt"
    return raw


def build_personaplex_ws_url(
    base_url: str,
    *,
    voice: str | None,
    text_prompt: str | None,
    audio_format: str = "pcm_f32",
) -> str:
    """Build moshi.server ``/api/chat`` WebSocket URL from an origin or full URL."""
    raw = (base_url or "").strip()
    if not raw:
        raise ValueError("PersonaPlex base_url is required")

    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()
    if scheme in ("http", "https", "ws", "wss"):
        if scheme == "http":
            scheme = "ws"
        elif scheme == "https":
            scheme = "wss"
    else:
        # Bare host:port or host
        raw = f"wss://{raw}"
        parsed = urlparse(raw)
        scheme = "wss"

    netloc = parsed.netloc or parsed.path.split("/")[0]
    path = parsed.path if parsed.netloc else ""
    if not path or path == "/":
        path = "/api/chat"
    elif not path.rstrip("/").endswith("/api/chat"):
        path = path.rstrip("/") + "/api/chat"

    query: dict[str, str] = {"audio_format": audio_format}
    query["voice_prompt"] = normalize_personaplex_voice(voice)
    if text_prompt and text_prompt.strip():
        query["text_prompt"] = text_prompt.strip()

    return urlunparse((scheme, netloc, path, "", urlencode(query), ""))


def pcm_s16le_to_f32le(pcm_s16: bytes) -> bytes:
    """Convert little-endian int16 PCM to float32 little-endian."""
    if not pcm_s16:
        return b""
    count = len(pcm_s16) // 2
    samples = struct.unpack(f"<{count}h", pcm_s16[: count * 2])
    return struct.pack(f"<{count}f", *(s / 32768.0 for s in samples))


def pcm_f32le_to_s16le(pcm_f32: bytes) -> bytes:
    """Convert little-endian float32 PCM to int16 little-endian."""
    if not pcm_f32:
        return b""
    count = len(pcm_f32) // 4
    floats = struct.unpack(f"<{count}f", pcm_f32[: count * 4])
    out = array.array("h")
    for sample in floats:
        clamped = max(-1.0, min(1.0, float(sample)))
        out.append(int(clamped * 32767.0))
    return out.tobytes()


def resample_pcm_s16le(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono int16 PCM. Uses audioop when available."""
    if not pcm or src_rate == dst_rate:
        return pcm
    if audioop is not None:
        try:
            converted, _ = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, None)
            return converted
        except Exception:
            pass
    # Linear interpolation fallback
    src = array.array("h")
    src.frombytes(pcm)
    if not src:
        return b""
    n_src = len(src)
    n_dst = max(1, int(round(n_src * dst_rate / src_rate)))
    dst = array.array("h")
    for i in range(n_dst):
        pos = i * (n_src - 1) / max(1, n_dst - 1)
        i0 = int(pos)
        i1 = min(i0 + 1, n_src - 1)
        frac = pos - i0
        sample = src[i0] * (1.0 - frac) + src[i1] * frac
        dst.append(int(sample))
    return dst.tobytes()


class DograhPersonaPlexRealtimeLLMService(LLMService):
    """PersonaPlex full-duplex realtime service for Dograh pipelines."""

    class Settings(LLMSettings):
        model: str | None = Field(default="nvidia/personaplex-7b-v1")
        voice: str | None = Field(default=_DEFAULT_VOICE)
        text_prompt: str | None = Field(default=None)
        system_instruction: str | None = Field(default=None)

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        settings: Settings | None = None,
        sample_rate: int | None = None,
        **kwargs,
    ):
        super().__init__(settings=settings or self.Settings(), **kwargs)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or None
        self._configured_sample_rate = sample_rate
        self._pipeline_in_sample_rate = sample_rate or 16000
        self._pipeline_out_sample_rate = sample_rate or 16000

        self._websocket = None
        self._receive_task: asyncio.Task | None = None
        self._user_is_muted = False
        self._connected = False
        self._handshake_done = False
        self._bot_is_speaking = False
        self._handled_initial_context = False
        self._reconnect_after_bot = False
        self._context: LLMContext | None = None
        self._text_buffer = ""
        self._connect_lock = asyncio.Lock()

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        self._warn_if_realtime_service_emits_no_turn_frames(emits_turn_frames=False)
        return LLMServiceMetadataFrame(
            service_name=self.name,
            is_realtime_service=True,
        )

    async def start(self, frame: StartFrame):
        # Defer WS connect until instruction/greeting is ready.
        await LLMService.start(self, frame)
        self._pipeline_in_sample_rate = (
            self._configured_sample_rate or frame.audio_in_sample_rate or 16000
        )
        self._pipeline_out_sample_rate = (
            self._configured_sample_rate or frame.audio_out_sample_rate or 16000
        )

    async def stop(self, frame: EndFrame):
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._disconnect()
        await super().cancel(frame)

    async def cleanup(self) -> None:
        await self._disconnect()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, UserMuteStartedFrame):
            self._user_is_muted = True
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, UserMuteStoppedFrame):
            self._user_is_muted = False
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSSpeakFrame):
            await self._handle_tts_speak(frame)
            return
        if isinstance(frame, LLMMessagesAppendFrame):
            # PersonaPlex has no mid-session text inject protocol in v1.
            logger.debug(f"{self}: ignoring LLMMessagesAppendFrame (no tool/text inject)")
            return
        if isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
            return
        if isinstance(frame, LLMContextFrame):
            self._context = frame.context
            # Pull system text if present for connect-time prompt
            instruction = self._extract_system_instruction(frame.context)
            if instruction:
                await self._maybe_set_instruction(instruction)
            await super().process_frame(frame, direction)
            return

        await super().process_frame(frame, direction)

    async def _update_settings(self, delta: LLMSettings) -> dict[str, Any]:
        changed = await super()._update_settings(delta)
        if "system_instruction" in changed:
            if self._connected:
                # Reconnect when bot is idle; otherwise flag for later.
                if self._bot_is_speaking:
                    self._reconnect_after_bot = True
                else:
                    await self._reconnect()
            elif self._settings.system_instruction:
                await self._connect()
        return changed

    async def _handle_tts_speak(self, frame: TTSSpeakFrame):
        if not self._connected:
            await self._connect(greeting_hint=frame.text)
        elif frame.text and frame.text.strip():
            logger.debug(
                f"{self}: TTSSpeakFrame after connect ignored "
                "(PersonaPlex owns speech output)"
            )
        self._handled_initial_context = True

    async def _maybe_set_instruction(self, instruction: str) -> None:
        if not instruction:
            return
        if self._settings.system_instruction == instruction:
            return
        await self._update_settings(self.Settings(system_instruction=instruction))

    def _extract_system_instruction(self, context: LLMContext | None) -> str | None:
        if context is None:
            return None
        messages = getattr(context, "messages", None) or []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text") or ""))
                    elif isinstance(part, str):
                        parts.append(part)
                joined = " ".join(p for p in parts if p).strip()
                if joined:
                    return joined
        return None

    def _resolve_text_prompt(self, greeting_hint: str | None = None) -> str | None:
        # Prefer runtime system instruction, then static config text_prompt.
        instruction = None
        if isinstance(self._settings.system_instruction, str):
            instruction = self._settings.system_instruction.strip() or None
        static = None
        if isinstance(getattr(self._settings, "text_prompt", None), str):
            static = self._settings.text_prompt.strip() or None

        parts = [p for p in (instruction, static) if p]
        if greeting_hint and greeting_hint.strip():
            parts.append(
                f"When the call starts, begin by saying exactly: {greeting_hint.strip()}"
            )
        if not parts:
            return None
        return "\n\n".join(parts)

    async def _connect(self, greeting_hint: str | None = None) -> None:
        async with self._connect_lock:
            if self._connected and self._websocket is not None:
                return
            text_prompt = self._resolve_text_prompt(greeting_hint)
            voice = getattr(self._settings, "voice", None) or _DEFAULT_VOICE
            url = build_personaplex_ws_url(
                self._base_url,
                voice=voice,
                text_prompt=text_prompt,
            )
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            logger.info(
                f"{self}: connecting to PersonaPlex at {urlparse(url).netloc} "
                f"(voice={normalize_personaplex_voice(voice)})"
            )
            try:
                # Disable client pings: first PersonaPlex handshake can take
                # minutes on cold start / cpu-offload while weights warm up.
                self._websocket = await websocket_connect(
                    url,
                    additional_headers=headers or None,
                    open_timeout=120,
                    ping_interval=None,
                    ping_timeout=None,
                    max_size=8 * 1024 * 1024,
                )
            except Exception as exc:
                self._websocket = None
                await self.push_error(
                    error_msg=(
                        "PersonaPlex unreachable. Start your GPU host (e.g. RunPod) "
                        f"and verify base_url. Detail: {exc}"
                    ),
                    exception=exc,
                )
                return

            self._connected = True
            self._handshake_done = False
            self._text_buffer = ""
            self._receive_task = self.create_task(self._receive_loop())

    async def _disconnect(self) -> None:
        self._connected = False
        self._handshake_done = False
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

    async def _reconnect(self) -> None:
        logger.info(f"{self}: reconnecting PersonaPlex for instruction change")
        await self._disconnect()
        await self._connect()

    async def _receive_loop(self) -> None:
        assert self._websocket is not None
        try:
            async for message in self._websocket:
                if isinstance(message, str):
                    logger.debug(f"{self}: ignoring text websocket message")
                    continue
                if not message:
                    continue
                kind = message[0]
                payload = message[1:]
                if kind == _KIND_HANDSHAKE:
                    self._handshake_done = True
                    logger.info(f"{self}: PersonaPlex handshake complete")
                elif kind == _KIND_TEXT:
                    await self._handle_text_token(payload)
                elif kind in (_KIND_PCM_F32, _KIND_OPUS):
                    if kind == _KIND_OPUS:
                        logger.warning(
                            f"{self}: received Opus frame but adapter uses pcm_f32; "
                            "check audio_format query param"
                        )
                        continue
                    await self._handle_pcm_audio(payload)
                else:
                    logger.debug(f"{self}: unknown frame kind={kind}")
        except ConnectionClosed as exc:
            if self._connected:
                logger.warning(f"{self}: PersonaPlex websocket closed: {exc}")
                await self.push_error(
                    error_msg="PersonaPlex connection closed. Restart the GPU host if the pod was stopped.",
                    exception=exc,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"{self}: receive loop error: {exc}")
            await self.push_error(
                error_msg=f"PersonaPlex receive error: {exc}",
                exception=exc,
            )
        finally:
            self._connected = False
            self._handshake_done = False

    async def _handle_text_token(self, payload: bytes) -> None:
        try:
            token = payload.decode("utf-8", errors="replace")
        except Exception:
            return
        if not token:
            return
        # Moshi streams tokens; treat sentence boundaries as finalized transcript chunks.
        self._text_buffer += token
        await self.push_frame(LLMTextFrame(text=token))
        if any(token.endswith(p) for p in (".", "?", "!", "\n")) or len(self._text_buffer) > 80:
            text = self._text_buffer.strip()
            self._text_buffer = ""
            if text:
                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id="",
                        timestamp=time_now_iso8601(),
                        finalized=True,
                    )
                )

    async def _handle_pcm_audio(self, payload: bytes) -> None:
        if not payload:
            return
        # float32 @ 24 kHz → int16 @ pipeline out rate
        pcm_s16_24k = pcm_f32le_to_s16le(payload)
        pcm_out = resample_pcm_s16le(
            pcm_s16_24k, _PERSONAPLEX_SAMPLE_RATE, self._pipeline_out_sample_rate
        )
        if not pcm_out:
            return
        if not self._bot_is_speaking:
            self._bot_is_speaking = True
            await self.push_frame(BotStartedSpeakingFrame())
            await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(
            TTSAudioRawFrame(
                audio=pcm_out,
                sample_rate=self._pipeline_out_sample_rate,
                num_channels=1,
            )
        )
        # Lightweight end-of-speech: schedule soft stop if silence-like
        # (full duplex: mark stop on short idle via background debounce)
        self._schedule_bot_stop_debounce()

    def _schedule_bot_stop_debounce(self) -> None:
        if getattr(self, "_bot_stop_task", None):
            self._bot_stop_task.cancel()

        async def _stop_later():
            try:
                await asyncio.sleep(0.45)
                if self._bot_is_speaking:
                    self._bot_is_speaking = False
                    await self.push_frame(LLMFullResponseEndFrame())
                    await self.push_frame(BotStoppedSpeakingFrame())
                    if self._reconnect_after_bot:
                        self._reconnect_after_bot = False
                        await self._reconnect()
            except asyncio.CancelledError:
                return

        self._bot_stop_task = asyncio.create_task(_stop_later())

    async def _send_user_audio(self, frame: InputAudioRawFrame) -> None:
        if self._user_is_muted or not self._connected or not self._handshake_done:
            return
        if self._websocket is None:
            return
        pcm = frame.audio
        if not pcm:
            return
        src_rate = frame.sample_rate or self._pipeline_in_sample_rate
        pcm_24k = resample_pcm_s16le(pcm, src_rate, _PERSONAPLEX_SAMPLE_RATE)
        pcm_f32 = pcm_s16le_to_f32le(pcm_24k)
        try:
            await self._websocket.send(bytes([_KIND_PCM_F32]) + pcm_f32)
        except ConnectionClosed:
            logger.warning(f"{self}: failed to send audio; connection closed")
        except Exception as exc:
            logger.warning(f"{self}: failed to send audio: {exc}")
