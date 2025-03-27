# trpc_backend.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from sqlalchemy import create_engine
import logging
from scrapers.rag import RAGSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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


@app.post("/trpc/query")
async def query_rag(request: QueryRequest) -> Dict:
    """
    Query the RAG system with a question about cryptocurrency
    """
    try:
        response = rag.generate_response(request.query, request.max_tokens)
        return {"result": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
                }
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
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
