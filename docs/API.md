# Voice RAG Assistant — API Reference

> Voice-powered AI assistant with knowledge base Q&A. Speak a question, get a spoken answer.

---

## Architecture

```
Audio Input (mic) → Whisper STT → Text Query
                                      ↓
                              ChromaDB Vector Search → Top-k Documents
                                      ↓
                              Claude LLM → Generated Answer
                                      ↓
                              ElevenLabs TTS → Audio Output (speaker)
```

---

## Python API

### SpeechToText

**Module:** `src/voice/speech_to_text.py`

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `(model_size: str = "base")` | — | Initialize Whisper model. Sizes: `tiny`, `base`, `small`, `medium`, `large` |
| `transcribe` | `(audio_file_path: str)` | `str` | Convert audio file to text |

**Example:**
```python
from src.voice.speech_to_text import SpeechToText

stt = SpeechToText(model_size="base")
text = stt.transcribe("recording.wav")
# "What time does the pool close?"
```

---

### KnowledgeBase

**Module:** `src/rag/knowledge_base.py`

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `(collection_name: str = "voice_assistant")` | — | Initialize ChromaDB + Claude backend |
| `add_documents` | `(texts: list[str], metadatas: list[dict] = None)` | — | Add documents to the knowledge base |
| `query` | `(question: str, n_results: int = 3)` | `dict` | Query KB and generate answer |
| `count` | `()` | `int` | Number of documents in the collection |

**Query Response:**
```json
{
  "answer": "The pool is open from 6 AM to 10 PM daily.",
  "sources": [
    "Pool hours are 6:00 AM to 10:00 PM, seven days a week."
  ]
}
```

**Example:**
```python
from src.rag.knowledge_base import KnowledgeBase

kb = KnowledgeBase()
kb.add_documents(
    texts=["Pool hours are 6:00 AM to 10:00 PM, seven days a week."],
    metadatas=[{"source": "hotel_info", "topic": "amenities"}]
)

result = kb.query("When does the pool close?")
print(result["answer"])
# "The pool closes at 10 PM daily."
```

---

### TextToSpeech

**Module:** `src/tts/text_to_speech.py`

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `(voice: str = "Rachel")` | — | Initialize ElevenLabs TTS |
| `speak` | `(text: str, output_path: str = None)` | `str` | Generate audio file, returns file path |
| `set_voice` | `(voice: str)` | — | Change voice |

**Available Voices:** Rachel, Drew, Sarah, Clyde, Domi, Dave, Fin, Bella, Antoni, Thomas, Charlie, Emily, Elli, Callum, Patrick, Harry, Liam, Dorothy, Josh, Arnold, Charlotte, Matilda

**Example:**
```python
from src.tts.text_to_speech import TextToSpeech

tts = TextToSpeech(voice="Rachel")
audio_path = tts.speak("The pool closes at 10 PM.")
# Returns path to generated .mp3 file
```

---

## Gradio Web UI

**Launch:** `python app.py` (default: `http://localhost:7860`)

### Tabs

| Tab | Function | Input | Output |
|-----|----------|-------|--------|
| Ask a Question | `process_voice(audio_path)` | Audio recording | Transcription + Answer text + Audio response |
| Add Documents | `add_document(text)` | Text string | Confirmation message |
| Settings | `change_voice(voice_name)` | Voice selection | Confirmation message |

### Full Pipeline

```python
def process_voice(audio_path: str) -> tuple[str, str, str]:
    """
    Complete voice Q&A pipeline.

    Args:
        audio_path: Path to recorded audio file

    Returns:
        tuple of (transcription, answer_text, audio_response_path)
    """
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Claude API key |
| `ELEVENLABS_API_KEY` | — | Required. ElevenLabs API key |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model size |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB storage path |

---

## Cost Estimate

| Component | Cost |
|-----------|------|
| Whisper STT | Free (runs locally) |
| ChromaDB | Free (runs locally) |
| Claude query | ~$0.003-0.01 |
| ElevenLabs TTS | ~$0.01-0.02 |
| **Total per query** | **~$0.01-0.03** |
