from openai import OpenAI
import numpy as np
from typing import List, Dict
from sqlalchemy import create_engine
# Import the similarity_search function
from generate_embeddings import similarity_search

# Initialize SQLite engine
sqlite_engine = create_engine("sqlite:///coindesk_embeddings.db")


class RAGSystem:
    def __init__(self, db_engine, embedding_type='combined', model="gpt-4-turbo-preview"):
        self.db_engine = db_engine
        self.embedding_type = embedding_type
        self.client = OpenAI()
        self.model = model

    def get_relevant_context(self, query: str, k: int = 3) -> List[Dict]:
        """Get most relevant articles using similarity search"""
        results = similarity_search(
            query, self.db_engine, self.embedding_type, k)
        return results

    def generate_response(self, query: str, max_tokens: int = 500) -> str:
        """Generate response using RAG"""
        # Get relevant context
        context_articles = self.get_relevant_context(query)

        # Format context for the prompt
        context_text = "\n\n".join([
            f"Title: {article['title']}\nContent: {article['content'][:500]}..."
            for article in context_articles
        ])

        # Create prompt
        prompt = f"""Based on the following articles about cryptocurrency and blockchain:

        {context_text}

        Please answer this question: {query}

        Provide a detailed response using information from the articles. If the articles don't contain relevant information, please state that."""

        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful expert in cryptocurrency and blockchain technology."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )

        return response.choices[0].message.content


# Usage
rag = RAGSystem(sqlite_engine)
response = rag.generate_response(
    "What are the latest developments in Bitcoin ETFs?")
print(response)
