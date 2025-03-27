# trpc_backend.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from sqlalchemy import create_engine
import logging
from scrapers.rag import RAGSystem
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from secure import Security, Headers
import time
from datetime import datetime, timedelta
import httpx

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="RAG API",
              description="Secure RAG system API",
              version="1.0.0")

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security headers middleware
security = Security()
headers = Headers()

# CORS middleware with strict settings
app.add_middleware(
    CORSMiddleware,
    # Replace with your frontend domain
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    max_age=3600,
)

# Security middleware


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    headers.secure_headers(response)
    return response

# API key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)


async def get_api_key(api_key_header: str = Depends(APIKeyHeader(name=API_KEY_NAME))):
    if api_key_header != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key_header

# Initialize database connection
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "scrapers", "coindesk_embeddings.db")
logger.info(f"Using database at: {db_path}")

if not os.path.exists(db_path):
    raise FileNotFoundError(f"Database file not found at {db_path}")

sqlite_engine = create_engine(f"sqlite:///{db_path}")

# Initialize RAG system
rag = RAGSystem(sqlite_engine)


class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 500


class ErrorResponse(BaseModel):
    detail: str
    timestamp: str

# Request validation middleware


@app.middleware("http")
async def validate_request(request: Request, call_next):
    # Check request size
    if request.headers.get("content-length") and \
       int(request.headers["content-length"]) > 1_000_000:  # 1MB limit
        return JSONResponse(
            status_code=413,
            content={"detail": "Request too large"}
        )

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.post("/trpc/query")
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MINUTE', '60')}/minute")
async def query_rag(
    request: Request,
    query_req: QueryRequest,
    api_key: str = Depends(get_api_key)
) -> Dict:
    """
    Query the RAG system with a question about cryptocurrency
    """
    try:
        # Validate max_tokens
        if query_req.max_tokens > 2000:
            raise HTTPException(
                status_code=400,
                detail="max_tokens cannot exceed 2000"
            )

        # Basic input sanitization
        query = query_req.query.strip()
        if not query:
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )

        if len(query) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Query too long (max 1000 characters)"
            )

        response = rag.generate_response(query, query_req.max_tokens)
        return {"result": response}

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return ErrorResponse(
            detail=str(e),
            timestamp=datetime.utcnow().isoformat()
        )


@app.get("/trpc/health")
async def health_check() -> Dict:
    """
    Health check endpoint that verifies database connection
    """
    try:
        # Verify database connection and content
        from scrapers.generate_embeddings import Article
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=sqlite_engine)
        session = Session()

        try:
            article_count = session.query(Article).count()
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
        return {
            "status": "unhealthy",
            "error": str(e),
            "database": {
                "connected": False,
                "path": db_path
            },
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
