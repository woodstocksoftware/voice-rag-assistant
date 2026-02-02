"""
Voice RAG Assistant - Gradio UI

Speak your question, get a spoken answer from the knowledge base.
"""

import gradio as gr
from src.voice.speech_to_text import SpeechToText
from src.rag.knowledge_base import KnowledgeBase
from src.tts.text_to_speech import TextToSpeech


# Initialize components
print("Loading Voice RAG Assistant...")
stt = SpeechToText()
kb = KnowledgeBase()
tts = TextToSpeech(voice="Rachel")
print(f"Knowledge base has {kb.count()} documents")
print("Ready!")


def process_voice(audio_path: str) -> tuple[str, str, str]:
    """
    Process voice input through the full pipeline.
    
    Returns: (transcription, answer_text, audio_path)
    """
    if audio_path is None:
        return "", "Please record a question first.", None
    
    # Step 1: Transcribe speech to text
    print(f"Transcribing audio: {audio_path}")
    transcription = stt.transcribe(audio_path)
    print(f"Transcription: {transcription}")
    
    if not transcription.strip():
        return "", "I couldn't hear anything. Please try again.", None
    
    # Step 2: Query the knowledge base
    print("Querying knowledge base...")
    result = kb.query(transcription)
    answer = result["answer"]
    print(f"Answer: {answer}")
    
    # Step 3: Convert answer to speech
    print("Generating speech...")
    audio_output = tts.speak(answer)
    print(f"Audio saved to: {audio_output}")
    
    return transcription, answer, audio_output


def add_document(text: str) -> str:
    """Add a document to the knowledge base."""
    if not text.strip():
        return "Please enter some text to add."
    
    kb.add_documents([text.strip()])
    return f"✅ Added document. Knowledge base now has {kb.count()} documents."


def change_voice(voice_name: str) -> str:
    """Change the TTS voice."""
    tts.set_voice(voice_name)
    return f"Voice changed to {voice_name}"


# Build the UI
with gr.Blocks(title="Voice RAG Assistant", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎙️ Voice RAG Assistant
    
    **Speak your question** and get a **spoken answer** from the knowledge base.
    
    Currently loaded with sample hotel information. Add your own documents below!
    """)
    
    with gr.Tabs():
        # Main voice interaction tab
        with gr.Tab("🎤 Ask a Question"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Record your question"
                    )
                    submit_btn = gr.Button("🚀 Get Answer", variant="primary")
                
                with gr.Column():
                    transcription_output = gr.Textbox(
                        label="What I heard",
                        interactive=False
                    )
                    answer_output = gr.Textbox(
                        label="Answer",
                        interactive=False
                    )
                    audio_output = gr.Audio(
                        label="🔊 Listen to answer",
                        type="filepath",
                        autoplay=True
                    )
            
            submit_btn.click(
                fn=process_voice,
                inputs=[audio_input],
                outputs=[transcription_output, answer_output, audio_output]
            )
        
        # Add documents tab
        with gr.Tab("📚 Add Documents"):
            gr.Markdown("""
            Add information to the knowledge base. The assistant will use this to answer questions.
            """)
            
            doc_input = gr.Textbox(
                label="Document text",
                placeholder="Enter information to add to the knowledge base...",
                lines=5
            )
            add_btn = gr.Button("➕ Add to Knowledge Base")
            add_result = gr.Textbox(label="Result", interactive=False)
            
            add_btn.click(
                fn=add_document,
                inputs=[doc_input],
                outputs=[add_result]
            )
            
            gr.Markdown(f"**Current knowledge base size:** {kb.count()} documents")
        
        # Settings tab
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### Voice Selection")
            
            voice_dropdown = gr.Dropdown(
                choices=[
                    "Rachel", "Drew", "Clyde", "Paul", "Domi", "Dave", 
                    "Fin", "Sarah", "Antoni", "Thomas", "Charlie", "George",
                    "Emily", "Elli", "Callum", "Patrick", "Harry", "Liam",
                    "Dorothy", "Josh", "Arnold", "Charlotte", "Alice", "Matilda"
                ],
                value="Rachel",
                label="Select voice"
            )
            voice_btn = gr.Button("Change Voice")
            voice_result = gr.Textbox(label="Result", interactive=False)
            
            voice_btn.click(
                fn=change_voice,
                inputs=[voice_dropdown],
                outputs=[voice_result]
            )
            
            gr.Markdown("""
            ### About
            
            This voice assistant uses:
            - **OpenAI Whisper** for speech-to-text
            - **ChromaDB** for vector search
            - **Claude** for answer generation
            - **ElevenLabs** for text-to-speech
            """)


if __name__ == "__main__":
    app.launch()
