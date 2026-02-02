"""
Speech-to-Text using local Whisper model.
Converts audio input to text transcription - runs locally, no API needed.
"""

import whisper


class SpeechToText:
    def __init__(self, model_size: str = "base"):
        """
        Initialize local Whisper model.
        
        Args:
            model_size: Model size - tiny, base, small, medium, large
                       - tiny: fastest, least accurate (~1GB VRAM)
                       - base: good balance (~1GB VRAM)
                       - small: better accuracy (~2GB VRAM)
                       - medium: high accuracy (~5GB VRAM)
                       - large: best accuracy (~10GB VRAM)
        """
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded!")
        
    def transcribe(self, audio_file_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_file_path: Path to audio file (mp3, wav, webm, etc.)
            
        Returns:
            Transcribed text
        """
        result = self.model.transcribe(audio_file_path)
        return result["text"].strip()


# Quick test
if __name__ == "__main__":
    stt = SpeechToText()
    print("Speech-to-Text module loaded successfully")
    print("Ready to transcribe audio files")
