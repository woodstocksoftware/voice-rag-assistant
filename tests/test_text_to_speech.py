"""
Tests for src.tts.text_to_speech.TextToSpeech.

The ElevenLabs client is mocked — no API key or network access required.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tts(mock_elevenlabs_client, voice="Rachel"):
    """
    Build a TextToSpeech instance with a mocked ElevenLabs client.
    ELEVENLABS_API_KEY is set by the autouse _fake_env fixture.
    """
    with patch("src.tts.text_to_speech.ElevenLabs") as MockElevenLabs:
        MockElevenLabs.return_value = mock_elevenlabs_client
        from src.tts.text_to_speech import TextToSpeech
        tts = TextToSpeech(voice=voice)
    return tts


# ---------------------------------------------------------------------------
# Tests — initialisation
# ---------------------------------------------------------------------------

class TestTextToSpeechInit:
    def test_default_voice_is_rachel(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs)
        assert tts.voice == "Rachel"

    def test_custom_voice(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs, voice="Drew")
        assert tts.voice == "Drew"

    def test_client_is_set(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs)
        assert tts.client is mock_elevenlabs

    def test_missing_api_key_raises(self, monkeypatch):
        """If ELEVENLABS_API_KEY is not set, constructor should raise ValueError."""
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
            from src.tts.text_to_speech import TextToSpeech
            TextToSpeech()


# ---------------------------------------------------------------------------
# Tests — speak
# ---------------------------------------------------------------------------

class TestSpeak:
    def test_speak_returns_file_path(self, mock_elevenlabs, tmp_path):
        tts = _make_tts(mock_elevenlabs)
        output_path = str(tmp_path / "output.mp3")

        result = tts.speak("Hello world", output_path=output_path)

        assert result == output_path

    def test_speak_writes_audio_bytes_to_file(self, mock_elevenlabs, tmp_path):
        mock_elevenlabs.text_to_speech.convert.return_value = [
            b"chunk1", b"chunk2", b"chunk3"
        ]
        tts = _make_tts(mock_elevenlabs)
        output_path = str(tmp_path / "output.mp3")

        tts.speak("Hello world", output_path=output_path)

        with open(output_path, "rb") as f:
            content = f.read()
        assert content == b"chunk1chunk2chunk3"

    def test_speak_uses_default_temp_path_when_none(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs)

        result = tts.speak("Hello")

        expected = os.path.join(tempfile.gettempdir(), "response.mp3")
        assert result == expected

    def test_speak_calls_elevenlabs_with_correct_params(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs, voice="Rachel")
        output_path = os.path.join(tempfile.gettempdir(), "test_speak.mp3")

        tts.speak("Test text", output_path=output_path)

        mock_elevenlabs.text_to_speech.convert.assert_called_once_with(
            text="Test text",
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel's voice ID
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

    def test_speak_uses_selected_voice_id(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs, voice="Drew")
        output_path = os.path.join(tempfile.gettempdir(), "test_drew.mp3")

        tts.speak("Test text", output_path=output_path)

        call_kwargs = mock_elevenlabs.text_to_speech.convert.call_args
        assert call_kwargs.kwargs["voice_id"] == "29vD33N1CtxCmqQRPOHJ"


# ---------------------------------------------------------------------------
# Tests — set_voice
# ---------------------------------------------------------------------------

class TestSetVoice:
    def test_set_voice_changes_voice(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs)
        assert tts.voice == "Rachel"

        tts.set_voice("Sarah")
        assert tts.voice == "Sarah"

    def test_set_voice_affects_subsequent_speak(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs)
        tts.set_voice("Sarah")
        output_path = os.path.join(tempfile.gettempdir(), "test_sarah.mp3")

        tts.speak("Hello", output_path=output_path)

        call_kwargs = mock_elevenlabs.text_to_speech.convert.call_args
        assert call_kwargs.kwargs["voice_id"] == "EXAVITQu4vr4xnSDxMaL"


# ---------------------------------------------------------------------------
# Tests — _get_voice_id
# ---------------------------------------------------------------------------

class TestGetVoiceId:
    def test_known_voices_return_correct_ids(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs)

        known = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM",
            "Drew": "29vD33N1CtxCmqQRPOHJ",
            "Clyde": "2EiwWnXFnvU5JabPnv8n",
            "Paul": "5Q0t7uMcjvnagumLfvZi",
            "Sarah": "EXAVITQu4vr4xnSDxMaL",
            "Charlie": "IKne3meq5aSn9XLyUdCD",
            "Emily": "LcfcDJNUP1GQjkzn1xUU",
            "Matilda": "XrExE9yKIg1WjnnlVkGX",
        }
        for name, expected_id in known.items():
            assert tts._get_voice_id(name) == expected_id

    def test_unknown_voice_falls_back_to_rachel(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs)
        # Unknown name should default to Rachel's ID
        assert tts._get_voice_id("NonexistentVoice") == "21m00Tcm4TlvDq8ikWAM"

    def test_all_24_voices_mapped(self, mock_elevenlabs):
        """The voice map should have exactly 24 entries."""
        tts = _make_tts(mock_elevenlabs)
        # Count unique IDs returned for all named voices
        all_voices = [
            "Rachel", "Drew", "Clyde", "Paul", "Domi", "Dave",
            "Fin", "Sarah", "Antoni", "Thomas", "Charlie", "George",
            "Emily", "Elli", "Callum", "Patrick", "Harry", "Liam",
            "Dorothy", "Josh", "Arnold", "Charlotte", "Alice", "Matilda",
        ]
        ids = [tts._get_voice_id(v) for v in all_voices]
        # All should return non-None values and be unique
        assert len(set(ids)) == 24

    def test_empty_voice_name_falls_back_to_rachel(self, mock_elevenlabs):
        tts = _make_tts(mock_elevenlabs)
        assert tts._get_voice_id("") == "21m00Tcm4TlvDq8ikWAM"
