#!/usr/bin/env python3
"""
Generate embeddings from Coindesk articles and store them in a database.

This script:
1. Reads article data from a CSV file
2. Generates embeddings using OpenAI's API
3. Stores the embeddings in a Neon Postgres database
4. Provides utility functions for querying the embeddings

Dependencies:
- pandas
- openai
- sqlalchemy
- psycopg2-binary
- python-dotenv
- tqdm

Install with:
pipenv install pandas openai sqlalchemy psycopg2-binary python-dotenv tqdm
"""

import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, MetaData, Table
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

# Load environment variables from .env file
load_dotenv()

# Configuration
DEFAULT_CSV_PATH = "coindesk_articles_20250315_183049.csv"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small
DEFAULT_BATCH_SIZE = 10
DEFAULT_CHUNK_SIZE = 8000  # Max tokens for embedding model

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please set it in a .env file or export it in your shell.")
    sys.exit(1)

client = openai.OpenAI(api_key=openai_api_key)

# Database setup
Base = declarative_base()


class Article(Base):
    """SQLAlchemy model for articles table."""
    __tablename__ = 'articles'

    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    url = Column(String(1000), unique=True)
    author = Column(String(255))
    date = Column(String(255))
    category = Column(String(255))
    summary = Column(Text)
    content = Column(Text)
    tags = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to embeddings
    embeddings = relationship(
        "Embedding", back_populates="article", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title[:30]}...', author='{self.author}')>"


class Embedding(Base):
    """SQLAlchemy model for embeddings table."""
    __tablename__ = 'embeddings'

    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('articles.id', ondelete='CASCADE'))
    model = Column(String(100), nullable=False)
    # 'title', 'content', 'combined'
    embedding_type = Column(String(50), nullable=False)
    vector_array = Column(ARRAY(Float))  # Store embeddings as Postgres array
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to article
    article = relationship("Article", back_populates="embeddings")

    def __repr__(self):
        return f"<Embedding(id={self.id}, article_id={self.article_id}, type='{self.embedding_type}')>"


def setup_database_connection() -> Any:
    """
    Set up connection to Neon Postgres database.

    Returns:
        SQLAlchemy engine
    """
    neon_postgres_url = os.getenv("NEON_POSTGRES_URL")
    if not neon_postgres_url:
        print("Error: NEON_POSTGRES_URL environment variable not set.")
        print("Please set it in a .env file or export it in your shell.")
        sys.exit(1)

    engine = create_engine(neon_postgres_url)

    # Create tables
    Base.metadata.create_all(engine)

    return engine


def read_articles_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Read articles from CSV file into a pandas DataFrame.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame containing article data
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Read {len(df)} articles from {csv_path}")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)


def generate_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    """
    Generate embedding for a text using OpenAI API.

    Args:
        text: Text to generate embedding for
        model: OpenAI model to use

    Returns:
        List of floats representing the embedding
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def process_articles(df: pd.DataFrame, model: str = DEFAULT_EMBEDDING_MODEL) -> List[Dict[str, Any]]:
    """
    Process articles and generate embeddings.

    Args:
        df: DataFrame containing articles
        model: OpenAI model to use for embeddings

    Returns:
        List of dictionaries containing article data and embeddings
    """
    articles_with_embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
        # Prepare text for embeddings
        title = str(row['title']).strip()
        content = str(row['content']).strip()
        combined = f"{title}\n\n{content}"

        # Generate embeddings
        title_embedding = generate_embedding(title, model)
        content_embedding = generate_embedding(content, model)
        combined_embedding = generate_embedding(combined, model)

        if not all([title_embedding, content_embedding, combined_embedding]):
            print(
                f"Warning: Failed to generate embeddings for article: {title}")
            continue

        # Prepare article data
        article_data = {
            'title': title,
            'url': row['url'],
            'author': row['author'],
            'date': row['date'],
            'category': row.get('category', ''),
            'summary': row.get('summary', ''),
            'content': content,
            'tags': row.get('tags', ''),
            'embeddings': [
                {
                    'type': 'title',
                    'vector': title_embedding,
                    'model': model
                },
                {
                    'type': 'content',
                    'vector': content_embedding,
                    'model': model
                },
                {
                    'type': 'combined',
                    'vector': combined_embedding,
                    'model': model
                }
            ]
        }
        articles_with_embeddings.append(article_data)

    return articles_with_embeddings


def store_in_database(articles_with_embeddings: List[Dict[str, Any]], engine: Any) -> None:
    """
    Store articles and embeddings in Postgres database.

    Args:
        articles_with_embeddings: List of article data with embeddings
        engine: SQLAlchemy engine
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        for article_data in tqdm(articles_with_embeddings, desc="Storing in database"):
            # Check if article already exists
            existing_article = session.query(Article).filter_by(
                url=article_data['url']).first()

            if existing_article:
                print(f"Article already exists: {article_data['title']}")
                continue

            # Create new article
            article = Article(
                title=article_data['title'],
                url=article_data['url'],
                author=article_data['author'],
                date=article_data['date'],
                category=article_data['category'],
                summary=article_data['summary'],
                content=article_data['content'],
                tags=article_data['tags']
            )

            # Add embeddings
            for emb_data in article_data['embeddings']:
                embedding = Embedding(
                    model=emb_data['model'],
                    embedding_type=emb_data['type'],
                    vector_array=emb_data['vector']
                )
                article.embeddings.append(embedding)

            session.add(article)
            session.commit()

        print(
            f"Successfully stored {len(articles_with_embeddings)} articles in database")

    except Exception as e:
        print(f"Error storing in database: {e}")
        session.rollback()
    finally:
        session.close()


def similarity_search(query_text: str, engine: Any, embedding_type: str = 'combined', top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar articles using embedding similarity.

    Args:
        query_text: Text to search for
        engine: SQLAlchemy engine
        embedding_type: Type of embedding to use for similarity search
        top_k: Number of results to return

    Returns:
        List of dictionaries containing similar articles
    """
    # Generate query embedding
    query_embedding = generate_embedding(query_text)
    if not query_embedding:
        return []

    # Convert to numpy array for calculations
    query_vector = np.array(query_embedding)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Get all embeddings of the specified type
        embeddings = session.query(Embedding, Article).join(Article).filter(
            Embedding.embedding_type == embedding_type).all()

        # Calculate similarities
        similarities = []
        for emb, article in embeddings:
            vector = np.array(emb.vector_array)
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((similarity, article))

        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:top_k]

        # Format results
        results = []
        for similarity, article in top_results:
            results.append({
                'title': article.title,
                'url': article.url,
                'author': article.author,
                'date': article.date,
                'category': article.category,
                'summary': article.summary,
                'similarity': float(similarity)
            })

        return results

    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate and store embeddings for articles')
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV_PATH,
                        help='Path to CSV file containing articles')
    parser.add_argument('--model', type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help='OpenAI model to use for embeddings')
    parser.add_argument('--search', type=str,
                        help='Search query (if provided, will perform similarity search)')
    args = parser.parse_args()

    # Set up database connection
    engine = setup_database_connection()

    if args.search:
        # Perform similarity search
        results = similarity_search(args.search, engine)
        print("\nSearch results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Author: {result['author']}")
            print(f"   Similarity: {result['similarity']:.4f}")
    else:
        # Process and store articles
        df = read_articles_from_csv(args.csv)
        articles_with_embeddings = process_articles(df, args.model)
        store_in_database(articles_with_embeddings, engine)


if __name__ == '__main__':
    main()
