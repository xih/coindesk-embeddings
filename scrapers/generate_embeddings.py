#!/usr/bin/env python3
"""
Generate embeddings from Coindesk articles and store them in databases.

This script:
1. Reads article data from a CSV file
2. Generates embeddings using OpenAI's API
3. Stores the embeddings in both a Neon Postgres database and a local SQLite database
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
DEFAULT_SQLITE_PATH = "coindesk_embeddings.db"
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

    # SQLite doesn't support arrays, so we'll store as JSON string
    vector_json = Column(Text)

    # For Postgres, we'll use the ARRAY type - but this is only defined in Postgres tables
    # vector_array is removed from here and added dynamically for Postgres

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to article
    article = relationship("Article", back_populates="embeddings")

    def __repr__(self):
        return f"<Embedding(id={self.id}, article_id={self.article_id}, type='{self.embedding_type}')>"


def setup_database_connections(postgres_url: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Set up connections to both SQLite and Postgres databases.

    Args:
        postgres_url: Connection URL for Postgres database

    Returns:
        Tuple of (sqlite_engine, postgres_engine)
    """
    # SQLite connection
    sqlite_engine = create_engine(f"sqlite:///{DEFAULT_SQLITE_PATH}")

    # Create tables in SQLite
    Base.metadata.create_all(sqlite_engine)

    # Postgres connection (if URL provided)
    postgres_engine = None
    if postgres_url:
        postgres_engine = create_engine(postgres_url)
    else:
        # Try to get from environment
        neon_postgres_url = os.getenv("NEON_POSTGRES_URL")
        if neon_postgres_url:
            postgres_engine = create_engine(neon_postgres_url)
        else:
            print("Warning: No Postgres URL provided. Only using SQLite database.")

    # Create tables in Postgres if available
    if postgres_engine:
        # For Postgres, we need to handle the ARRAY type differently
        # Create tables with SQLAlchemy core instead of ORM
        metadata = MetaData()

        articles = Table(
            'articles', metadata,
            Column('id', Integer, primary_key=True),
            Column('title', String(500), nullable=False),
            Column('url', String(1000), unique=True),
            Column('author', String(255)),
            Column('date', String(255)),
            Column('category', String(255)),
            Column('summary', Text),
            Column('content', Text),
            Column('tags', String(500)),
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        embeddings = Table(
            'embeddings', metadata,
            Column('id', Integer, primary_key=True),
            Column('article_id', Integer, ForeignKey(
                'articles.id', ondelete='CASCADE')),
            Column('model', String(100), nullable=False),
            Column('embedding_type', String(50), nullable=False),
            Column('vector_json', Text),  # Keep this for compatibility
            Column('vector_array', ARRAY(Float)),  # Postgres-specific column
            Column('created_at', DateTime, default=datetime.utcnow)
        )

        metadata.create_all(postgres_engine)

    return sqlite_engine, postgres_engine


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
        model: OpenAI embedding model to use

    Returns:
        List of floats representing the embedding vector
    """
    try:
        # Truncate text if it's too long
        if len(text) > DEFAULT_CHUNK_SIZE:
            text = text[:DEFAULT_CHUNK_SIZE]

        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def process_articles(df: pd.DataFrame, model: str = DEFAULT_EMBEDDING_MODEL) -> List[Dict[str, Any]]:
    """
    Process articles and generate embeddings.

    Args:
        df: DataFrame containing article data
        model: OpenAI embedding model to use

    Returns:
        List of dictionaries containing article data and embeddings
    """
    results = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
        article_data = row.to_dict()

        # Prepare text for embeddings
        title = article_data.get('Title', '')
        content = article_data.get('Content', '')
        summary = article_data.get('Summary', '')

        # Skip if no meaningful content
        if title == '' and (content == '' or content == 'No content available'):
            print(f"Skipping article {i} due to lack of content")
            continue

        # Generate embeddings for different parts
        title_embedding = generate_embedding(title, model) if title else []

        # Use content if available, otherwise use summary
        if content and content != 'No content available':
            content_embedding = generate_embedding(content, model)
        elif summary and summary != 'No summary available':
            content_embedding = generate_embedding(summary, model)
        else:
            content_embedding = []

        # Generate combined embedding if both title and content are available
        combined_text = f"{title}\n\n{content if content != 'No content available' else summary}"
        combined_embedding = generate_embedding(combined_text, model)

        # Add embeddings to article data
        article_data['title_embedding'] = title_embedding
        article_data['content_embedding'] = content_embedding
        article_data['combined_embedding'] = combined_embedding

        results.append(article_data)

        # Sleep to avoid rate limits
        time.sleep(0.5)

    return results


def store_in_sqlite(articles_with_embeddings: List[Dict[str, Any]], engine: Any) -> None:
    """
    Store articles and embeddings in SQLite database.

    Args:
        articles_with_embeddings: List of dictionaries containing article data and embeddings
        engine: SQLAlchemy engine for SQLite
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        for article_data in tqdm(articles_with_embeddings, desc="Storing in SQLite"):
            # Create article record
            article = Article(
                title=article_data.get('Title', ''),
                url=article_data.get('URL', ''),
                author=article_data.get('Author', ''),
                date=article_data.get('Date', ''),
                category=article_data.get('Category', ''),
                summary=article_data.get('Summary', ''),
                content=article_data.get('Content', ''),
                tags=article_data.get('Tags', '')
            )
            session.add(article)
            session.flush()  # Get the article ID

            # Create embedding records
            if article_data.get('title_embedding'):
                title_embedding = Embedding(
                    article_id=article.id,
                    model=DEFAULT_EMBEDDING_MODEL,
                    embedding_type='title',
                    vector_json=json.dumps(article_data['title_embedding'])
                )
                session.add(title_embedding)

            if article_data.get('content_embedding'):
                content_embedding = Embedding(
                    article_id=article.id,
                    model=DEFAULT_EMBEDDING_MODEL,
                    embedding_type='content',
                    vector_json=json.dumps(article_data['content_embedding'])
                )
                session.add(content_embedding)

            if article_data.get('combined_embedding'):
                combined_embedding = Embedding(
                    article_id=article.id,
                    model=DEFAULT_EMBEDDING_MODEL,
                    embedding_type='combined',
                    vector_json=json.dumps(article_data['combined_embedding'])
                )
                session.add(combined_embedding)

        session.commit()
        print(
            f"Successfully stored {len(articles_with_embeddings)} articles in SQLite database")
    except Exception as e:
        session.rollback()
        print(f"Error storing in SQLite: {e}")
    finally:
        session.close()


def store_in_postgres(articles_with_embeddings: List[Dict[str, Any]], engine: Any) -> None:
    """
    Store articles and embeddings in Postgres database.

    Args:
        articles_with_embeddings: List of dictionaries containing article data and embeddings
        engine: SQLAlchemy engine for Postgres
    """
    if not engine:
        print("Skipping Postgres storage as no connection is available")
        return

    # Use SQLAlchemy Core for better performance with arrays
    metadata = MetaData()
    articles_table = Table('articles', metadata, autoload_with=engine)
    embeddings_table = Table('embeddings', metadata, autoload_with=engine)

    connection = engine.connect()
    transaction = connection.begin()

    try:
        for article_data in tqdm(articles_with_embeddings, desc="Storing in Postgres"):
            # Insert article
            article_result = connection.execute(
                articles_table.insert().values(
                    title=article_data.get('Title', ''),
                    url=article_data.get('URL', ''),
                    author=article_data.get('Author', ''),
                    date=article_data.get('Date', ''),
                    category=article_data.get('Category', ''),
                    summary=article_data.get('Summary', ''),
                    content=article_data.get('Content', ''),
                    tags=article_data.get('Tags', ''),
                    created_at=datetime.utcnow()
                ).returning(articles_table.c.id)
            )
            article_id = article_result.fetchone()[0]

            # Insert embeddings
            if article_data.get('title_embedding'):
                connection.execute(
                    embeddings_table.insert().values(
                        article_id=article_id,
                        model=DEFAULT_EMBEDDING_MODEL,
                        embedding_type='title',
                        vector_json=json.dumps(
                            article_data['title_embedding']),
                        vector_array=article_data['title_embedding'],
                        created_at=datetime.utcnow()
                    )
                )

            if article_data.get('content_embedding'):
                connection.execute(
                    embeddings_table.insert().values(
                        article_id=article_id,
                        model=DEFAULT_EMBEDDING_MODEL,
                        embedding_type='content',
                        vector_json=json.dumps(
                            article_data['content_embedding']),
                        vector_array=article_data['content_embedding'],
                        created_at=datetime.utcnow()
                    )
                )

            if article_data.get('combined_embedding'):
                connection.execute(
                    embeddings_table.insert().values(
                        article_id=article_id,
                        model=DEFAULT_EMBEDDING_MODEL,
                        embedding_type='combined',
                        vector_json=json.dumps(
                            article_data['combined_embedding']),
                        vector_array=article_data['combined_embedding'],
                        created_at=datetime.utcnow()
                    )
                )

        transaction.commit()
        print(
            f"Successfully stored {len(articles_with_embeddings)} articles in Postgres database")
    except Exception as e:
        transaction.rollback()
        print(f"Error storing in Postgres: {e}")
    finally:
        connection.close()


def similarity_search(query_text: str, engine: Any, embedding_type: str = 'combined', top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform similarity search using embeddings.

    Args:
        query_text: Query text to search for
        engine: SQLAlchemy engine (SQLite or Postgres)
        embedding_type: Type of embedding to use ('title', 'content', or 'combined')
        top_k: Number of results to return

    Returns:
        List of dictionaries containing article data and similarity scores
    """
    # Generate embedding for query
    query_embedding = generate_embedding(query_text)

    # Convert to numpy array for easier calculations
    query_embedding_np = np.array(query_embedding)

    # Get all embeddings of the specified type
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Get all articles and their embeddings
        results = session.query(Article, Embedding).join(
            Embedding, Article.id == Embedding.article_id
        ).filter(
            Embedding.embedding_type == embedding_type
        ).all()

        similarities = []

        for article, embedding in results:
            # Get embedding vector
            if engine.url.drivername == 'sqlite':
                # For SQLite, parse the JSON string
                vector = np.array(json.loads(embedding.vector_json))
            else:
                # For Postgres, use the array directly
                vector = np.array(embedding.vector_array)

            # Calculate cosine similarity
            similarity = np.dot(query_embedding_np, vector) / (
                np.linalg.norm(query_embedding_np) * np.linalg.norm(vector)
            )

            similarities.append({
                'article_id': article.id,
                'title': article.title,
                'url': article.url,
                'author': article.author,
                'date': article.date,
                'category': article.category,
                'summary': article.summary,
                'content': article.content,
                'tags': article.tags,
                'similarity': float(similarity)
            })

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Return top k results
        return similarities[:top_k]

    finally:
        session.close()


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Generate embeddings from Coindesk articles')
    parser.add_argument('--csv', type=str,
                        default=DEFAULT_CSV_PATH, help='Path to CSV file')
    parser.add_argument('--postgres-url', type=str,
                        help='Postgres connection URL')
    parser.add_argument(
        '--model', type=str, default=DEFAULT_EMBEDDING_MODEL, help='OpenAI embedding model')
    parser.add_argument('--search', type=str,
                        help='Perform similarity search with this query')
    args = parser.parse_args()

    # Set up database connections
    sqlite_engine, postgres_engine = setup_database_connections(
        args.postgres_url)

    # If search query provided, perform search and exit
    if args.search:
        print(f"Performing similarity search for: {args.search}")
        results = similarity_search(args.search, sqlite_engine)
        print("\nSearch Results:")
        for i, result in enumerate(results):
            print(
                f"\n{i+1}. {result['title']} (Score: {result['similarity']:.4f})")
            print(f"   Author: {result['author']}")
            print(f"   Category: {result['category']}")
            print(f"   URL: {result['url']}")
        return

    # Read articles from CSV
    csv_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), args.csv)
    df = read_articles_from_csv(csv_path)

    # Process articles and generate embeddings
    articles_with_embeddings = process_articles(df, args.model)

    # Store in SQLite
    store_in_sqlite(articles_with_embeddings, sqlite_engine)

    # Store in Postgres (if available)
    if postgres_engine:
        store_in_postgres(articles_with_embeddings, postgres_engine)

    print("Done!")


if __name__ == "__main__":
    main()
