"""
Shared fixtures for voice-rag-assistant tests.

All external APIs (Anthropic Claude, OpenAI Whisper, ElevenLabs) are mocked
so tests can run without API keys or network access.
"""

import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Environment: ensure no real API keys are needed
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fake_env(monkeypatch):
    """Set fake API keys so constructors don't blow up."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test_fake_key")


# ---------------------------------------------------------------------------
# ChromaDB: use a temporary directory for each test so tests are isolated
# ---------------------------------------------------------------------------

@pytest.fixture()
def chroma_tmp_dir():
    """Create and clean up a temporary directory for ChromaDB storage."""
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Anthropic Claude mock
# ---------------------------------------------------------------------------

def _make_claude_response(text: str = "This is a mock answer from Claude."):
    """Build a mock Anthropic Messages response."""
    content_block = MagicMock()
    content_block.text = text

    response = MagicMock()
    response.content = [content_block]
    return response


@pytest.fixture()
def mock_anthropic():
    """
    Patch anthropic.Anthropic so no real API call is made.

    Returns the mock client instance so tests can inspect calls or
    override return values.
    """
    with patch("src.rag.knowledge_base.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_claude_response()
        MockAnthropic.return_value = mock_client
        yield mock_client


# ---------------------------------------------------------------------------
# Whisper mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_whisper():
    """
    Patch whisper.load_model so no real model is downloaded.

    Returns the mock model instance.
    """
    with patch("src.voice.speech_to_text.whisper") as mock_whisper_module:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " Hello, what time is check-in? "}
        mock_whisper_module.load_model.return_value = mock_model
        yield mock_model


# ---------------------------------------------------------------------------
# ElevenLabs mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_elevenlabs():
    """
    Patch elevenlabs.ElevenLabs so no real API call is made.

    Returns the mock client instance.
    """
    with patch("src.tts.text_to_speech.ElevenLabs") as MockElevenLabs:
        mock_client = MagicMock()
        # text_to_speech.convert returns an iterable of audio bytes
        mock_client.text_to_speech.convert.return_value = [b"fake-audio-bytes"]
        MockElevenLabs.return_value = mock_client
        yield mock_client


# ---------------------------------------------------------------------------
# Sample hotel documents (matches app.py seed data)
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    "Our hotel check-in time is 3 PM and check-out time is 11 AM. Early check-in may be available upon request.",
    "The swimming pool is located on the 5th floor and is open from 6 AM to 10 PM daily.",
    "Room service is available 24 hours. You can order by pressing 0 on your room phone.",
    "Free WiFi is available throughout the hotel. The password is provided at check-in.",
    "The fitness center is on the 3rd floor, open 24 hours for hotel guests.",
]

SAMPLE_METADATAS = [
    {"source": "check-in policy"},
    {"source": "pool info"},
    {"source": "room service"},
    {"source": "wifi info"},
    {"source": "fitness center"},
]
