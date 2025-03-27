# trpc_backend.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import APIKeyHeader  # Commented out security imports
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from sqlalchemy import create_engine
import logging
from scrapers.rag import RAGSystem
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
from datetime import datetime, timedelta
import httpx
import platform
import sys
import traceback

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("Starting application...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Platform: {platform.platform()}")
logger.info(f"Working directory: {os.getcwd()}")

try:
    # Initialize rate limiter
    logger.info("Initializing rate limiter...")
    limiter = Limiter(key_func=get_remote_address)

    app = FastAPI(title="RAG API",
                  description="Secure RAG system API",
                  version="1.0.0")

    # Add rate limiter to app
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiter configured")

    # CORS middleware with permissive settings for testing
    logger.info("Configuring CORS...")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for testing
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
    logger.info("CORS configured")

    # Initialize database connection
    logger.info("Initializing database connection...")
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "scrapers", "coindesk_embeddings.db")
    logger.info(f"Database path: {db_path}")

    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}")
        raise FileNotFoundError(f"Database file not found at {db_path}")

    sqlite_engine = create_engine(f"sqlite:///{db_path}")
    logger.info("Database connection established")

    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag = RAGSystem(sqlite_engine)
    logger.info("RAG system initialized")

    class QueryRequest(BaseModel):
        query: str
        max_tokens: Optional[int] = 500

    class ErrorResponse(BaseModel):
        detail: str
        timestamp: str

    @app.post("/trpc/query")
    @limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MINUTE', '60')}/minute")
    async def query_rag(
        request: Request,
        query_req: QueryRequest,
    ) -> Dict:
        """
        Query the RAG system with a question about cryptocurrency
        """
        try:
            logger.info(f"Processing query request: {query_req.query[:50]}...")

            # Validate max_tokens
            if query_req.max_tokens > 2000:
                logger.warning(f"max_tokens too high: {query_req.max_tokens}")
                raise HTTPException(
                    status_code=400,
                    detail="max_tokens cannot exceed 2000"
                )

            # Basic input sanitization
            query = query_req.query.strip()
            if not query:
                logger.warning("Empty query received")
                raise HTTPException(
                    status_code=400,
                    detail="Query cannot be empty"
                )

            if len(query) > 1000:
                logger.warning(f"Query too long: {len(query)} characters")
                raise HTTPException(
                    status_code=400,
                    detail="Query too long (max 1000 characters)"
                )

            logger.info("Generating response...")
            response = rag.generate_response(query, query_req.max_tokens)
            logger.info("Response generated successfully")
            return {"result": response}

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ErrorResponse(
                detail=str(e),
                timestamp=datetime.utcnow().isoformat()
            )

    @app.get("/trpc/health")
    async def health_check() -> Dict:
        """
        Health check endpoint that verifies database connection
        """
        logger.info("Processing health check request")
        try:
            # Verify database connection and content
            from scrapers.generate_embeddings import Article
            from sqlalchemy.orm import sessionmaker

            Session = sessionmaker(bind=sqlite_engine)
            session = Session()

            try:
                article_count = session.query(Article).count()
                logger.info(
                    f"Health check successful. Found {article_count} articles")
                return {
                    "status": "healthy",
                    "database": {
                        "connected": True,
                        "article_count": article_count,
                        "path": db_path
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "database": {
                    "connected": False,
                    "path": db_path
                },
                "timestamp": datetime.utcnow().isoformat()
            }

    @app.get("/trpc/environment")
    async def environment_check() -> Dict:
        """
        Check the deployment environment
        """
        logger.info("Processing environment check request")
        try:
            is_docker = os.path.exists(
                '/.dockerenv') or os.path.exists('/run/.containerenv')

            env_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "is_docker": is_docker,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "environment": {
                    "RENDER": os.getenv("RENDER"),
                    "RENDER_SERVICE_ID": os.getenv("RENDER_SERVICE_ID"),
                    "PYTHON_VERSION": os.getenv("PYTHON_VERSION")
                },
                "filesystem": {
                    "dockerfile_exists": os.path.exists("/app/Dockerfile"),
                    "requirements_exists": os.path.exists("/app/requirements.txt"),
                    "working_directory": os.getcwd()
                },
                "imported_packages": {
                    "fastapi": "fastapi" in sys.modules,
                    "uvicorn": "uvicorn" in sys.modules
                }
            }
            logger.info(f"Environment check successful: {env_info}")
            return env_info

        except Exception as e:
            logger.error(f"Environment check failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

except Exception as e:
    logger.critical(f"Failed to initialize application: {str(e)}")
    logger.critical(f"Traceback: {traceback.format_exc()}")
    raise

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
