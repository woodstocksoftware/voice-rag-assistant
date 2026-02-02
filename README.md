# Voice RAG Assistant

A voice-powered AI assistant that answers questions from a knowledge base. Speak your question, get a spoken answer.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Whisper](https://img.shields.io/badge/STT-Whisper-green)
![Claude](https://img.shields.io/badge/LLM-Claude-blueviolet)
![ElevenLabs](https://img.shields.io/badge/TTS-ElevenLabs-orange)

## How It Works
```
🎤 You speak
     │
     ▼
┌─────────────┐
│   Whisper   │  ← Speech-to-Text (local)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ChromaDB   │  ← Vector search for relevant docs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Claude    │  ← Generate conversational answer
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ ElevenLabs  │  ← Text-to-Speech
└──────┬──────┘
       │
       ▼
🔊 Assistant speaks
```

## Demo

The assistant comes pre-loaded with sample hotel information:
- Check-in/check-out times
- Pool location and hours
- Room service
- WiFi information
- Fitness center

Ask questions like:
- "What time is check-in?"
- "Where is the swimming pool?"
- "How do I order room service?"

## Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **Speech-to-Text** | Whisper (local) | Runs on CPU, no API needed |
| **Vector Database** | ChromaDB | Local, persistent storage |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2 |
| **LLM** | Claude | Conversational answers |
| **Text-to-Speech** | ElevenLabs | Natural-sounding voices |
| **UI** | Gradio | Web interface with audio |

## Setup
```bash
# Clone
git clone https://github.com/woodstocksoftware/voice-rag-assistant.git
cd voice-rag-assistant

# Create environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install ffmpeg (required for Whisper)
brew install ffmpeg  # macOS
# sudo apt install ffmpeg  # Ubuntu

# Set API keys
export ANTHROPIC_API_KEY="your-key"
export ELEVENLABS_API_KEY="your-key"

# Run
python app.py
```

Open **http://localhost:7860** and start talking!

## Project Structure
```
voice-rag-assistant/
├── app.py                      # Gradio UI
├── src/
│   ├── voice/
│   │   └── speech_to_text.py   # Whisper transcription
│   ├── rag/
│   │   └── knowledge_base.py   # ChromaDB + Claude
│   └── tts/
│       └── text_to_speech.py   # ElevenLabs TTS
├── data/
│   └── chroma/                 # Vector database storage
└── requirements.txt
```

## Adding Your Own Knowledge

Use the "Add Documents" tab in the UI, or programmatically:
```python
from src.rag.knowledge_base import KnowledgeBase

kb = KnowledgeBase()
kb.add_documents([
    "Your custom information here.",
    "More information about your domain.",
])
```

## Voice Options

ElevenLabs offers many voices. Change in the Settings tab or:
```python
from src.tts.text_to_speech import TextToSpeech

tts = TextToSpeech(voice="Drew")  # Male voice
tts = TextToSpeech(voice="Sarah")  # Different female voice
```

## Use Cases

### Hospitality
- Hotel concierge answering guest questions
- Restaurant information and reservations

### Education
- Voice tutoring assistant
- Interactive learning companion
- Accessibility for visually impaired students

### Enterprise
- Internal knowledge base with voice interface
- Hands-free documentation lookup

## Cost

| Service | Cost |
|---------|------|
| Whisper | Free (runs locally) |
| ChromaDB | Free (runs locally) |
| Claude | ~$0.01-0.03 per query |
| ElevenLabs | Free tier: 10,000 chars/month |

## License

MIT

---

Built by [Jim Williams](https://linkedin.com/in/woodstocksoftware) | [GitHub](https://github.com/woodstocksoftware)
