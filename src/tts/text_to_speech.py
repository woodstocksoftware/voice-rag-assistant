"""
Text-to-Speech using ElevenLabs API.
Converts text responses to natural-sounding speech.
"""

import os
import tempfile
from elevenlabs import ElevenLabs


class TextToSpeech:
    def __init__(self, voice: str = "Rachel"):
        """
        Initialize TTS with ElevenLabs.
        
        Args:
            voice: Voice to use (Rachel, Drew, Sarah, etc.)
        """
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")
        self.client = ElevenLabs(api_key=api_key)
        self.voice = voice
        
    def speak(self, text: str, output_path: str = None) -> str:
        """
        Convert text to speech and save to file.
        
        Args:
            text: Text to convert to speech
            output_path: Where to save the audio (default: temp file)
            
        Returns:
            Path to the generated audio file
        """
        if output_path is None:
            # Use system temp directory for Gradio compatibility
            output_path = os.path.join(tempfile.gettempdir(), "response.mp3")
        
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self._get_voice_id(self.voice),
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        # Write audio to file
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        
        return output_path
    
    def _get_voice_id(self, voice_name: str) -> str:
        """Get voice ID from name."""
        voices = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM",
            "Drew": "29vD33N1CtxCmqQRPOHJ",
            "Clyde": "2EiwWnXFnvU5JabPnv8n",
            "Paul": "5Q0t7uMcjvnagumLfvZi",
            "Domi": "AZnzlk1XvdvUeBnXmlld",
            "Dave": "CYw3kZ02Hs0563khs1Fj",
            "Fin": "D38z5RcWu1voky8WS1ja",
            "Sarah": "EXAVITQu4vr4xnSDxMaL",
            "Antoni": "ErXwobaYiN019PkySvjV",
            "Thomas": "GBv7mTt0atIp3Br8iCZE",
            "Charlie": "IKne3meq5aSn9XLyUdCD",
            "George": "JBFqnCBsd6RMkjVDRZzb",
            "Emily": "LcfcDJNUP1GQjkzn1xUU",
            "Elli": "MF3mGyEYCl7XYWbV9V6O",
            "Callum": "N2lVS1w4EtoT3dr4eOWO",
            "Patrick": "ODq5zmih8GrVes37Dizd",
            "Harry": "SOYHLrjzK2X1ezoPC6cr",
            "Liam": "TX3LPaxmHKxFdv7VOQHJ",
            "Dorothy": "ThT5KcBeYPX3keUQqHPh",
            "Josh": "TxGEqnHWrfWFTfGW9XjX",
            "Arnold": "VR6AewLTigWG4xSOukaG",
            "Charlotte": "XB0fDUnXU5powFXDhCwa",
            "Alice": "Xb7hH8MSUJpSbSDYk0k2",
            "Matilda": "XrExE9yKIg1WjnnlVkGX",
        }
        return voices.get(voice_name, voices["Rachel"])
    
    def set_voice(self, voice: str):
        """Change the voice."""
        self.voice = voice


# Quick test
if __name__ == "__main__":
    tts = TextToSpeech(voice="Rachel")
    print("Text-to-Speech module loaded successfully")
    output = tts.speak("Hello! I'm your voice assistant.")
    print(f"Generated audio at: {output}")
