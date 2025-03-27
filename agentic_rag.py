#!/usr/bin/env python3
"""
Enhanced Agentic RAG system with improved capabilities for cryptocurrency research
"""

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.cache import InMemoryCache
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime, timedelta
import json
from functools import lru_cache
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import spacy
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class EnhancedAgenticRAG:
    def __init__(self, db_engine, model="gpt-4-turbo-preview", cache_size: int = 1000):
        self.db_engine = db_engine
        self.llm = ChatOpenAI(model=model)
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 interactions
        )
        self.cache = InMemoryCache()
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.setup_vectorstore()
        self.setup_agent()

    def setup_vectorstore(self):
        """Initialize vector store for faster similarity search"""
        try:
            Session = sessionmaker(bind=self.db_engine)
            session = Session()

            # Get all articles and their embeddings
            query = text("""
                SELECT a.id, a.title, a.content, a.published_at, a.category,
                       e.vector_json
                FROM articles a
                JOIN embeddings e ON a.id = e.article_id
                WHERE e.embedding_type = 'combined'
            """)

            results = session.execute(query)
            documents = []
            metadata = []

            for row in results:
                documents.append(row.content)
                metadata.append({
                    'id': row.id,
                    'title': row.title,
                    'published_at': row.published_at,
                    'category': row.category,
                })

            session.close()

            if documents:
                self.vectorstore = FAISS.from_texts(
                    documents,
                    self.embeddings,
                    metadatas=metadata
                )
                logger.info(
                    f"Vectorstore initialized with {len(documents)} documents")
            else:
                logger.warning("No documents found in database")

        except Exception as e:
            logger.error(f"Error setting up vectorstore: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Enhanced similarity search with retry logic and caching"""
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized")

            results = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k
            )

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    def analyze_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """Analyze sentiment of articles using TextBlob"""
        try:
            sentiments = {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0
            }

            for text in texts:
                analysis = TextBlob(text)
                polarity = analysis.sentiment.polarity

                if polarity > 0.1:
                    sentiments["positive"] += 1
                elif polarity < -0.1:
                    sentiments["negative"] += 1
                else:
                    sentiments["neutral"] += 1

            total = len(texts)
            if total > 0:
                sentiments = {k: v/total for k, v in sentiments.items()}

            return sentiments

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"error": str(e)}

    def cluster_articles(self, articles: List[Dict]) -> Dict[str, List[Dict]]:
        """Cluster articles by topic using TF-IDF and KMeans"""
        try:
            if not articles:
                return {"clusters": []}

            # Extract text content
            texts = [article['content'] for article in articles]

            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Determine optimal number of clusters (max 5)
            n_clusters = min(5, len(articles))

            # Perform clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42
            )
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            # Group articles by cluster
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label] = clusters[label] + [articles[idx]]

            # Get top terms for each cluster
            feature_names = vectorizer.get_feature_names_out()
            for label, cluster in clusters.items():
                centroid = kmeans.cluster_centers_[label]
                top_indices = centroid.argsort()[-5:][::-1]
                top_terms = [feature_names[i] for i in top_indices]
                clusters[label] = {
                    'articles': cluster,
                    'top_terms': top_terms
                }

            return {"clusters": clusters}

        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            return {"error": str(e)}

    def extract_key_metrics(self, articles: List[Dict]) -> Dict[str, Any]:
        """Extract key metrics and entities from articles"""
        try:
            metrics = {
                "total_articles": len(articles),
                "date_range": {
                    "start": None,
                    "end": None
                },
                "top_entities": [],
                "key_statistics": {}
            }

            if not articles:
                return metrics

            # Extract dates
            dates = [
                datetime.fromisoformat(art['metadata']['published_at'])
                for art in articles
                if 'metadata' in art and 'published_at' in art['metadata']
            ]

            if dates:
                metrics["date_range"] = {
                    "start": min(dates).isoformat(),
                    "end": max(dates).isoformat()
                }

            # Extract entities
            all_entities = []
            for article in articles:
                doc = nlp(article['content'])
                entities = [
                    (ent.text, ent.label_)
                    for ent in doc.ents
                    if ent.label_ in ['ORG', 'PERSON', 'GPE', 'MONEY', 'PERCENT']
                ]
                all_entities.extend(entities)

            # Count and sort entities
            entity_counter = Counter(all_entities)
            metrics["top_entities"] = [
                {"text": ent[0], "label": ent[1], "count": count}
                for (ent, count) in entity_counter.most_common(10)
            ]

            # Calculate key statistics
            metrics["key_statistics"] = {
                "avg_length": np.mean([len(art['content'].split()) for art in articles]),
                "categories": Counter([
                    art['metadata']['category']
                    for art in articles
                    if 'metadata' in art and 'category' in art['metadata']
                ])
            }

            return metrics

        except Exception as e:
            logger.error(f"Error extracting metrics: {str(e)}")
            return {"error": str(e)}

    def setup_agent(self):
        """Set up the enhanced research agent with additional tools"""
        tools = [
            Tool(
                name="Search Articles",
                func=self.similarity_search,
                description="Search through cryptocurrency articles with semantic similarity"
            ),
            Tool(
                name="Analyze Trends",
                func=self.analyze_trends,
                description="Analyze trends across multiple articles with temporal analysis"
            ),
            Tool(
                name="Sentiment Analysis",
                func=self.analyze_sentiment,
                description="Analyze sentiment of cryptocurrency articles"
            ),
            Tool(
                name="Cluster Topics",
                func=self.cluster_articles,
                description="Cluster articles by topic similarity"
            ),
            Tool(
                name="Extract Metrics",
                func=self.extract_key_metrics,
                description="Extract key metrics and statistics from articles"
            )
        ]

        prompt = PromptTemplate.from_template("""
        You are an advanced cryptocurrency research agent with access to historical article data.
        Your goal is to provide detailed, data-driven analysis using the available tools.
        
        Available tools:
        {tools}
        
        Tool names: {tool_names}
        
        Context from previous interactions:
        {chat_history}
        
        Current Question: {query}
        
        Think step by step:
        1. What specific information do you need to answer this query?
        2. Which combination of tools would provide the most comprehensive analysis?
        3. How can you validate and cross-reference the information?
        4. What potential biases or limitations should be considered?
        5. How can you present the findings in a structured way?
        
        Remember to:
        - Provide confidence scores for your findings
        - Include relevant metrics and trends
        - Consider temporal aspects of the data
        - Highlight any contradictory information
        - Suggest follow-up areas for investigation
        
        {agent_scratchpad}
        """)

        self.agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            memory=self.memory,
            verbose=True
        )

    def analyze_trends(self, query: str, timeframe: str = "1m") -> Dict:
        """Enhanced trend analysis with temporal aspects"""
        try:
            # Get articles within timeframe
            results = self.similarity_search(query, top_k=20)

            # Analyze temporal patterns
            temporal_analysis = self._analyze_temporal_patterns(results)

            # Get sentiment over time
            sentiment_analysis = self.analyze_sentiment(
                [r['content'] for r in results])

            # Cluster topics
            topic_clusters = self.cluster_articles(results)

            # Extract key metrics
            metrics = self.extract_key_metrics(results)

            return {
                "trend_summary": {
                    "temporal_patterns": temporal_analysis,
                    "sentiment_trends": sentiment_analysis,
                    "topic_clusters": topic_clusters,
                    "key_metrics": metrics,
                    "confidence_score": self._calculate_confidence(results)
                }
            }
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return {"error": str(e)}

    def _analyze_temporal_patterns(self, results: List[Dict]) -> Dict:
        """Analyze temporal patterns in the data"""
        try:
            if not results:
                return {"temporal_patterns": "No data available"}

            # Convert to DataFrame for time series analysis
            df = pd.DataFrame([
                {
                    'date': datetime.fromisoformat(r['metadata']['published_at']),
                    'content': r['content'],
                    'category': r['metadata']['category']
                }
                for r in results
                if 'metadata' in r and 'published_at' in r['metadata']
            ])

            if df.empty:
                return {"temporal_patterns": "No temporal data available"}

            # Sort by date
            df = df.sort_values('date')

            # Daily article counts
            daily_counts = df.groupby(df['date'].dt.date).size()

            # Category trends
            category_trends = df.groupby([
                df['date'].dt.date,
                'category'
            ]).size().unstack(fill_value=0)

            # Calculate moving averages
            if len(daily_counts) > 3:
                moving_avg = daily_counts.rolling(window=3).mean()
            else:
                moving_avg = daily_counts

            return {
                "temporal_patterns": {
                    "daily_counts": daily_counts.to_dict(),
                    "moving_average": moving_avg.to_dict(),
                    "category_trends": category_trends.to_dict(),
                    "date_range": {
                        "start": df['date'].min().isoformat(),
                        "end": df['date'].max().isoformat()
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error in temporal analysis: {str(e)}")
            return {"error": str(e)}

    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score based on data quality and quantity"""
        try:
            if not results:
                return 0.0

            # Factors affecting confidence:
            # 1. Number of articles (more is better)
            # 2. Recency of articles
            # 3. Similarity scores
            # 4. Source diversity

            n_articles = len(results)
            max_articles = 20  # Normalize against expected maximum

            # Calculate article score (0.4 weight)
            article_score = min(n_articles / max_articles, 1.0) * 0.4

            # Calculate recency score (0.3 weight)
            now = datetime.now()
            age_scores = []
            for result in results:
                if 'metadata' in result and 'published_at' in result['metadata']:
                    pub_date = datetime.fromisoformat(
                        result['metadata']['published_at'])
                    age_days = (now - pub_date).days
                    age_score = max(0, 1 - (age_days / 30))  # Newer is better
                    age_scores.append(age_score)

            recency_score = (np.mean(age_scores) if age_scores else 0.5) * 0.3

            # Calculate similarity score (0.2 weight)
            similarity_scores = [
                result.get('similarity_score', 0.5)
                for result in results
            ]
            similarity_score = (np.mean(similarity_scores)
                                if similarity_scores else 0.5) * 0.2

            # Calculate source diversity (0.1 weight)
            sources = set(
                result['metadata'].get('category', '')
                for result in results
                if 'metadata' in result
            )
            diversity_score = min(len(sources) / 5, 1.0) * 0.1

            total_confidence = article_score + recency_score + \
                similarity_score + diversity_score

            return round(total_confidence, 2)

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5  # Default middle confidence

    @lru_cache(maxsize=100)
    def research(self, query: str) -> Dict[str, Any]:
        """Conduct research using the agent with caching"""
        try:
            with get_openai_callback() as cb:
                response = self.agent_executor.invoke({
                    "query": query,
                    "chat_history": self.memory.chat_memory.messages
                })

                # Structure the response
                structured_response = {
                    "answer": response["output"],
                    "confidence_score": self._calculate_confidence(
                        response.get("intermediate_steps", [])
                    ),
                    "sources": self._extract_sources(response),
                    "metadata": {
                        "token_usage": cb.total_tokens,
                        "cost": cb.total_cost,
                        "timestamp": datetime.now().isoformat()
                    }
                }

                return structured_response
        except Exception as e:
            logger.error(f"Error in research: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _extract_sources(self, response: Dict) -> List[Dict]:
        """Extract and format sources used in the research"""
        try:
            sources = []
            if "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    if isinstance(step[1], list):
                        for item in step[1]:
                            if isinstance(item, dict) and 'metadata' in item:
                                sources.append({
                                    'title': item['metadata'].get('title', ''),
                                    'published_at': item['metadata'].get('published_at', ''),
                                    'category': item['metadata'].get('category', ''),
                                    'similarity_score': item.get('similarity_score', 0.0)
                                })

            # Remove duplicates while preserving order
            seen = set()
            unique_sources = []
            for source in sources:
                source_tuple = tuple(source.items())
                if source_tuple not in seen:
                    seen.add(source_tuple)
                    unique_sources.append(source)

            return unique_sources

        except Exception as e:
            logger.error(f"Error extracting sources: {str(e)}")
            return []


# Usage example
if __name__ == "__main__":
    from sqlalchemy import create_engine

    # Initialize database connection
    db_path = "scrapers/coindesk_embeddings.db"
    sqlite_engine = create_engine(f"sqlite:///{db_path}")

    # Create agent
    agent_rag = EnhancedAgenticRAG(sqlite_engine)

    # Example research query
    research_result = agent_rag.research(
        "What are the major trends in DeFi over the past month, "
        "focusing on TVL changes and new protocol launches?"
    )

    # Print results
    print(json.dumps(research_result, indent=2))
