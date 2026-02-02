"""
Simple RAG knowledge base using ChromaDB.
Reuses patterns from your RAG Documentation Assistant.
"""

import os
from anthropic import Anthropic
import chromadb
from chromadb.utils import embedding_functions


class KnowledgeBase:
    def __init__(self, collection_name: str = "voice_assistant"):
        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma")
        
        # Use sentence-transformers for embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Claude for generation
        self.claude = Anthropic()
        
    def add_documents(self, texts: list[str], metadatas: list[dict] = None):
        """Add documents to the knowledge base."""
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
            
        ids = [f"doc_{self.collection.count() + i}" for i in range(len(texts))]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(texts)} documents to knowledge base")
        
    def query(self, question: str, n_results: int = 3) -> dict:
        """
        Query the knowledge base and generate an answer.
        
        Returns dict with 'answer' and 'sources'.
        """
        # Retrieve relevant documents
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        # Build context from results
        context_parts = []
        sources = []
        
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0]
            )):
                context_parts.append(f"[Source {i+1}]: {doc}")
                sources.append(metadata.get('source', f'Source {i+1}'))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with Claude
        if not context:
            return {
                "answer": "I don't have any information about that in my knowledge base.",
                "sources": []
            }
        
        response = self.claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system="""You are a helpful voice assistant. Answer questions based on the provided context.
            
Rules:
- Be conversational and natural - your response will be spoken aloud
- Keep answers concise (2-4 sentences ideal for voice)
- If the context doesn't contain the answer, say so briefly
- Don't use markdown formatting, bullet points, or numbered lists
- Don't say "according to the source" - just answer naturally""",
            messages=[{
                "role": "user",
                "content": f"""Context:
{context}

Question: {question}

Provide a brief, conversational answer suitable for voice output."""
            }]
        )
        
        return {
            "answer": response.content[0].text,
            "sources": sources
        }
    
    def count(self) -> int:
        """Return number of documents in the knowledge base."""
        return self.collection.count()


# Quick test
if __name__ == "__main__":
    kb = KnowledgeBase()
    print(f"Knowledge base initialized with {kb.count()} documents")
    
    # Add some test data if empty
    if kb.count() == 0:
        kb.add_documents([
            "Our hotel check-in time is 3 PM and check-out time is 11 AM. Early check-in may be available upon request.",
            "The swimming pool is located on the 5th floor and is open from 6 AM to 10 PM daily.",
            "Room service is available 24 hours. You can order by pressing 0 on your room phone.",
            "Free WiFi is available throughout the hotel. The password is provided at check-in.",
            "The fitness center is on the 3rd floor, open 24 hours for hotel guests."
        ], metadatas=[
            {"source": "check-in policy"},
            {"source": "pool info"},
            {"source": "room service"},
            {"source": "wifi info"},
            {"source": "fitness center"}
        ])
    
    # Test query
    result = kb.query("What time can I check in?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Sources: {result['sources']}")
