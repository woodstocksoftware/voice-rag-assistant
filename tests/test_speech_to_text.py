"""
Tests for src.voice.speech_to_text.SpeechToText.

The Whisper model is mocked -- no model download or GPU required.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stt(mock_model, model_size="base"):
    """
    Build a SpeechToText instance with a mocked whisper module.
    Returns (stt_instance, mock_whisper_module) so tests can inspect calls.
    """
    with patch("src.voice.speech_to_text.whisper") as mock_module:
        mock_module.load_model.return_value = mock_model
        from src.voice.speech_to_text import SpeechToText
        stt = SpeechToText(model_size=model_size)
    return stt, mock_module


def _make_mock_model(transcribe_text=" Hello, what time is check-in? "):
    """Create a mock whisper model with configurable transcription output."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": transcribe_text}
    return mock_model


# ---------------------------------------------------------------------------
# Tests -- initialisation
# ---------------------------------------------------------------------------

class TestSpeechToTextInit:
    def test_default_model_size_is_base(self):
        """Default constructor uses 'base' model."""
        mock_model = _make_mock_model()
        _, mock_module = _make_stt(mock_model)
        mock_module.load_model.assert_called_once_with("base")

    def test_custom_model_size(self):
        """Constructor passes the requested model size."""
        mock_model = _make_mock_model()
        _, mock_module = _make_stt(mock_model, model_size="tiny")
        mock_module.load_model.assert_called_once_with("tiny")

    def test_model_attribute_set(self):
        """After init, stt.model is the loaded model."""
        mock_model = _make_mock_model()
        stt, _ = _make_stt(mock_model)
        assert stt.model is mock_model


# ---------------------------------------------------------------------------
# Tests -- transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_transcribe_returns_stripped_text(self):
        """transcribe() should strip whitespace from the result."""
        mock_model = _make_mock_model("  Hello, world!  ")
        stt, _ = _make_stt(mock_model)

        result = stt.transcribe("/tmp/test_audio.wav")
        assert result == "Hello, world!"

    def test_transcribe_passes_file_path_to_model(self):
        """The audio file path is forwarded to model.transcribe()."""
        mock_model = _make_mock_model("test")
        stt, _ = _make_stt(mock_model)

        stt.transcribe("/tmp/recording.mp3")
        mock_model.transcribe.assert_called_once_with("/tmp/recording.mp3")

    def test_transcribe_empty_audio(self):
        """When Whisper returns empty text, transcribe returns empty string."""
        mock_model = _make_mock_model("   ")
        stt, _ = _make_stt(mock_model)

        result = stt.transcribe("/tmp/silence.wav")
        assert result == ""

    def test_transcribe_long_text(self):
        """Whisper can return long transcriptions."""
        long_text = "This is a really long question about " + "things " * 50
        mock_model = _make_mock_model(long_text)
        stt, _ = _make_stt(mock_model)

        result = stt.transcribe("/tmp/long_audio.wav")
        assert result == long_text.strip()

    def test_transcribe_unicode_text(self):
        """Whisper may return unicode characters."""
        mock_model = _make_mock_model(" Bonjour, comment allez-vous? ")
        stt, _ = _make_stt(mock_model)

        result = stt.transcribe("/tmp/french.wav")
        assert result == "Bonjour, comment allez-vous?"

    def test_transcribe_multiple_calls(self):
        """Multiple calls to transcribe should work independently."""
        mock_model = _make_mock_model("First call")
        stt, _ = _make_stt(mock_model)

        result1 = stt.transcribe("/tmp/audio1.wav")
        assert result1 == "First call"

        mock_model.transcribe.return_value = {"text": "Second call"}
        result2 = stt.transcribe("/tmp/audio2.wav")
        assert result2 == "Second call"
        assert mock_model.transcribe.call_count == 2


# ---------------------------------------------------------------------------
# Tests -- model sizes
# ---------------------------------------------------------------------------

class TestModelSizes:
    @pytest.mark.parametrize("size", ["tiny", "base", "small", "medium", "large"])
    def test_all_valid_model_sizes(self, size):
        """Each supported model size should be passed through to load_model."""
        mock_model = _make_mock_model()
        _, mock_module = _make_stt(mock_model, model_size=size)
        mock_module.load_model.assert_called_once_with(size)
