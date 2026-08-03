"""Unit tests for NVIDIA PersonaPlex realtime provider (no live GPU)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration.check_validity import UserConfigurationValidator
from api.services.configuration.registry import (
    PersonaPlexRealtimeLLMConfiguration,
    ServiceProviders,
)
from api.services.pipecat.realtime.personaplex_realtime import (
    DograhPersonaPlexRealtimeLLMService,
    build_personaplex_ws_url,
    normalize_personaplex_voice,
    pcm_f32le_to_s16le,
    pcm_s16le_to_f32le,
    resample_pcm_s16le,
)
from api.services.pipecat.run_pipeline import _create_realtime_user_turn_config
from api.services.pipecat.service_factory import create_realtime_llm_service
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.turns.user_start.vad_user_turn_start_strategy import (
    VADUserTurnStartStrategy,
)


def test_personaplex_configuration_defaults():
    config = PersonaPlexRealtimeLLMConfiguration(
        provider=ServiceProviders.PERSONAPLEX_REALTIME,
        base_url="wss://example-8998.proxy.runpod.net",
    )
    assert config.model == "nvidia/personaplex-7b-v1"
    assert config.voice == "NATF2"
    assert config.api_key is None
    assert config.text_prompt is None


def test_normalize_personaplex_voice_adds_pt_suffix():
    assert normalize_personaplex_voice("NATF2") == "NATF2.pt"
    assert normalize_personaplex_voice("NATM1.pt") == "NATM1.pt"
    assert normalize_personaplex_voice(None) == "NATF2.pt"


def test_build_personaplex_ws_url_from_origin():
    url = build_personaplex_ws_url(
        "wss://pod-8998.proxy.runpod.net",
        voice="NATF2",
        text_prompt="You are a friendly assistant.",
    )
    assert url.startswith("wss://pod-8998.proxy.runpod.net/api/chat?")
    assert "voice_prompt=NATF2.pt" in url
    assert "audio_format=pcm_f32" in url
    assert "text_prompt=" in url


def test_build_personaplex_ws_url_https_to_wss():
    url = build_personaplex_ws_url(
        "https://pod-8998.proxy.runpod.net",
        voice="NATM0",
        text_prompt=None,
    )
    assert url.startswith("wss://")
    assert "voice_prompt=NATM0.pt" in url


def test_pcm_roundtrip_s16_f32():
    original = b"\x00\x00\xff\x7f\x00\x80"  # 0, max, min int16
    f32 = pcm_s16le_to_f32le(original)
    assert len(f32) == 12
    back = pcm_f32le_to_s16le(f32)
    assert len(back) == 6


def test_resample_identity_same_rate():
    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    assert resample_pcm_s16le(pcm, 16000, 16000) == pcm


def test_personaplex_uses_local_vad_with_interruptions():
    strategies, vad_analyzer = _create_realtime_user_turn_config(
        ServiceProviders.PERSONAPLEX_REALTIME.value
    )
    assert isinstance(vad_analyzer, SileroVADAnalyzer)
    assert len(strategies.start) == 1
    assert isinstance(strategies.start[0], VADUserTurnStartStrategy)
    assert strategies.start[0]._enable_interruptions is True


def test_factory_creates_personaplex_service():
    effective_config = EffectiveAIModelConfiguration(
        is_realtime=True,
        realtime=PersonaPlexRealtimeLLMConfiguration(
            provider=ServiceProviders.PERSONAPLEX_REALTIME,
            base_url="wss://pod-8998.proxy.runpod.net",
            api_key=None,
            model="nvidia/personaplex-7b-v1",
            voice="NATF2",
            text_prompt="You enjoy a good conversation.",
        ),
    )
    with patch(
        "api.services.pipecat.service_factory._validate_runtime_service_url"
    ) as mock_validate:
        service = create_realtime_llm_service(
            effective_config,
            audio_config=SimpleNamespace(pipeline_sample_rate=16000),
        )
    mock_validate.assert_called_once()
    assert isinstance(service, DograhPersonaPlexRealtimeLLMService)
    assert service._base_url == "wss://pod-8998.proxy.runpod.net"
    assert service._settings.voice == "NATF2"


def test_factory_requires_base_url():
    config = MagicMock()
    config.realtime = MagicMock()
    config.realtime.provider = ServiceProviders.PERSONAPLEX_REALTIME.value
    config.realtime.model = "nvidia/personaplex-7b-v1"
    config.realtime.api_key = None
    config.realtime.voice = "NATF2"
    config.realtime.base_url = ""
    config.realtime.text_prompt = None

    with pytest.raises(HTTPException) as exc:
        create_realtime_llm_service(
            config, audio_config=SimpleNamespace(pipeline_sample_rate=16000)
        )
    assert exc.value.status_code == 400
    assert "base_url" in str(exc.value.detail).lower()


@pytest.mark.asyncio
async def test_validator_accepts_personaplex_with_base_url_only():
    from api.services.configuration.registry import OpenAILLMService

    validator = UserConfigurationValidator()
    configuration = EffectiveAIModelConfiguration(
        is_realtime=True,
        llm=OpenAILLMService(
            provider=ServiceProviders.OPENAI,
            api_key="sk-test",
            model="gpt-4.1-mini",
        ),
        realtime=PersonaPlexRealtimeLLMConfiguration(
            provider=ServiceProviders.PERSONAPLEX_REALTIME,
            base_url="wss://pod-8998.proxy.runpod.net",
            api_key=None,
        ),
    )
    with patch.object(
        validator, "_check_openai_api_key", return_value=True
    ), patch(
        "api.services.configuration.check_validity.validate_user_configured_service_url"
    ):
        result = await validator.validate(configuration)
    assert result["status"][0]["message"] == "ok"


@pytest.mark.asyncio
async def test_validator_rejects_personaplex_without_base_url():
    from api.services.configuration.registry import OpenAILLMService

    validator = UserConfigurationValidator()
    # Pydantic will also require base_url on the config class — exercise the
    # runtime check path with a MagicMock-like object via _validate_service.
    service = MagicMock()
    service.provider = ServiceProviders.PERSONAPLEX_REALTIME.value
    service.base_url = ""
    service.api_key = None
    errors = validator._validate_service(service, "realtime")
    assert errors
    assert "base_url" in errors[0]["message"].lower()
