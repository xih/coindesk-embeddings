from openai import OpenAI
import numpy as np
import os
import json
from typing import List, Dict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, db_engine, embedding_type='combined', model="gpt-4-turbo-preview"):
        self.db_engine = db_engine
        self.embedding_type = embedding_type
        self.client = OpenAI()
        self.model = model
        self.Session = sessionmaker(bind=self.db_engine)

    def get_relevant_context(self, query: str, k: int = 3) -> List[Dict]:
        """Get most relevant articles using similarity search"""
        try:
            session = self.Session()

            # Get the query embedding from OpenAI
            query_embedding = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                encoding_format="float"
            ).data[0].embedding

            # Convert embedding to numpy array
            query_vector = np.array(query_embedding)

            # SQL query to get articles and their embeddings
            sql = text("""
                SELECT a.id, a.title, a.content, a.category, a.date,
                       e.vector_json
                FROM articles a
                JOIN embeddings e ON a.id = e.article_id
                WHERE e.embedding_type = :embedding_type
            """)

            results = session.execute(
                sql, {"embedding_type": self.embedding_type})

            # Calculate similarities and store results
            similarities = []
            for row in results:
                vector = np.array(json.loads(row.vector_json))
                similarity = np.dot(query_vector, vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(vector)
                )
                similarities.append({
                    'id': row.id,
                    'title': row.title,
                    'content': row.content,
                    'category': row.category,
                    'date': row.date,
                    'similarity': float(similarity)
                })

            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_k_results = similarities[:k]

            logger.info(
                f"Found {len(top_k_results)} relevant articles for query: {query[:50]}...")
            return top_k_results

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
        finally:
            session.close()

    def generate_response(self, query: str, max_tokens: int = 500) -> str:
        """Generate response using RAG"""
        try:
            # Get relevant context
            context_articles = self.get_relevant_context(query)

            if not context_articles:
                logger.warning("No relevant articles found for query")
                return "I apologize, but I couldn't find any relevant articles to answer your question accurately."

            logger.info(f"Found {len(context_articles)} relevant articles")

            # Format context for the prompt
            context_text = "\n\n".join([
                f"Title: {article['title']}\nDate: {article['date']}\nCategory: {article['category']}\nContent: {article['content'][:500]}..."
                for article in context_articles
            ])

            # Create prompt
            prompt = f"""Based on the following articles about cryptocurrency and blockchain:

            {context_text}

            Please answer this question: {query}

            Provide a detailed response using information from the articles. If the articles don't contain relevant information, please state that.
            
            Include specific details, dates, and figures when available. If the information is time-sensitive, mention when the source articles were published."""

            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful expert in cryptocurrency and blockchain technology. Provide accurate, well-sourced information based on the provided articles."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but an error occurred while processing your request: {str(e)}"


# Only run this when the file is executed directly, not when imported
if __name__ == "__main__":
    # Initialize database connection
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "coindesk_embeddings.db")
    print(f"Using database at: {db_path}")

    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        exit(1)

    sqlite_engine = create_engine(f"sqlite:///{db_path}")

    # Check if database has been populated
    Session = sessionmaker(bind=sqlite_engine)
    session = Session()

    try:
        from generate_embeddings import Article
        article_count = session.query(Article).count()

        if article_count == 0:
            print("Warning: No articles found in the database!")
            print("Run the generate_embeddings.py script first to populate the database.")
            print("Example: python generate_embeddings.py")
        else:
            print(f"Found {article_count} articles in the database.")
            # Usage example
            rag = RAGSystem(sqlite_engine)
            response = rag.generate_response(
                "What are the latest developments in Bitcoin ETFs?")
            print("\nResponse:")
            print(response)

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        session.close()
