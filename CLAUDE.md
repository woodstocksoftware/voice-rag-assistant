# CLAUDE.md — Voice RAG Assistant

> **Purpose:** Voice-powered RAG — speak a question, get a spoken answer from a knowledge base
> **Owner:** Jim Williams - Woodstock Software LLC
> **Repo:** woodstocksoftware/voice-rag-assistant (public)

---

## Tech Stack

- Python 3.12
- OpenAI Whisper (local speech-to-text, no API key needed)
- ChromaDB (vector database, persistent local storage)
- sentence-transformers (all-MiniLM-L6-v2 embeddings)
- Anthropic Claude Sonnet (RAG generation)
- ElevenLabs (text-to-speech API)
- Gradio (web UI)
- python-dotenv

## Project Structure

```
voice-rag-assistant/
├── app.py                         # Gradio UI entry point (3 tabs)
├── requirements.txt
├── src/
│   ├── voice/
│   │   └── speech_to_text.py      # Whisper STT wrapper
│   ├── rag/
│   │   └── knowledge_base.py      # ChromaDB + Claude RAG
│   └── tts/
│       └── text_to_speech.py      # ElevenLabs TTS wrapper
├── data/chroma/                   # Persistent vector store
├── LICENSE                        # MIT
└── README.md
```

## How to Run

```bash
cd /Users/james/projects/voice-rag-assistant
source venv/bin/activate
export ANTHROPIC_API_KEY="sk-ant-..."
export ELEVENLABS_API_KEY="sk_..."
python app.py
# Opens http://localhost:7860
```

**System dependency:** `brew install ffmpeg` (required for Whisper audio processing)

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes | Claude API for RAG generation |
| `ELEVENLABS_API_KEY` | Yes | ElevenLabs text-to-speech |

## Pipeline

```
Microphone → Whisper (local) → text
  → ChromaDB search (top 3) → Claude generation → text answer
  → ElevenLabs TTS → MP3 → Gradio playback
```

### Key Components

1. **SpeechToText** — Loads Whisper model locally (configurable: tiny/base/small/medium/large). No API key needed.
2. **KnowledgeBase** — ChromaDB with cosine similarity. Auto-seeds 5 hotel sample docs if empty. Claude Sonnet generates answers with max 500 tokens.
3. **TextToSpeech** — ElevenLabs API, eleven_multilingual_v2 model, 24 voice options. Outputs temp MP3 files.

### Gradio UI (3 tabs)
- **Ask Question**: Voice recording → full pipeline → audio + text output
- **Add Documents**: Upload text to knowledge base
- **Settings**: Voice selection, knowledge base status

## Testing

No formal test suite. Each module has `__main__` blocks for manual testing:
```bash
python src/voice/speech_to_text.py       # Test Whisper
python src/rag/knowledge_base.py         # Test ChromaDB + Claude
python src/tts/text_to_speech.py         # Test ElevenLabs
```

## Key Patterns

- **Local-first STT**: Whisper runs on CPU, no API needed (~1GB RAM for base model)
- **Persistent vectors**: ChromaDB saves to `./data/chroma/`
- **Modular design**: STT, RAG, TTS are independent — can swap any component
- **Cosine similarity**: all-MiniLM-L6-v2 embeddings (384 dimensions)
- **Voice-optimized prompts**: Claude system prompt tuned for concise, conversational responses

## Cost

- Whisper: Free (local)
- ChromaDB: Free (local)
- Claude: ~$0.01-0.03/query
- ElevenLabs: Free tier 10,000 chars/month

## What's Missing

- [ ] Tests (pytest)
- [ ] CI workflow (.github/workflows/)
- [ ] pyproject.toml
