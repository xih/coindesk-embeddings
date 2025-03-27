# Secure RAG API System

A secure RAG (Retrieval-Augmented Generation) API system for cryptocurrency research, built with FastAPI and OpenAI.

## Security Features

- API Key Authentication
- Rate Limiting
- CORS Protection
- Request Size Validation
- Security Headers
- Input Sanitization
- Error Handling
- Request Logging
- DDoS Protection (via Render)

## Deployment on Render

1. Fork/Clone this repository

2. Create a new Web Service on Render:

   - Go to https://dashboard.render.com
   - Click "New +" and select "Web Service"
   - Connect your repository
   - Choose a name for your service

3. Configuration:

   - Environment: Python 3.9
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn trpc_backend:app --host 0.0.0.0 --port $PORT`

4. Environment Variables:
   Set the following in Render dashboard:

   ```
   OPENAI_API_KEY=your_openai_api_key
   API_KEY=your_secure_api_key
   RATE_LIMIT_PER_MINUTE=60
   ```

5. Add the following environment variables to your deployment:
   - `PYTHON_VERSION`: 3.9.0
   - `PORT`: Let Render set this automatically

## API Usage

1. Authentication:
   Include your API key in requests:

   ```bash
   curl -X POST https://your-api.onrender.com/trpc/query \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the latest developments in Bitcoin ETFs?"}'
   ```

2. Rate Limits:

   - 60 requests per minute per IP
   - Adjust `RATE_LIMIT_PER_MINUTE` in environment variables

3. Request Limits:
   - Maximum query length: 1000 characters
   - Maximum request size: 1MB
   - Maximum tokens: 2000

## Security Best Practices

1. API Keys:

   - Generate strong API keys
   - Rotate keys regularly
   - Never expose keys in client-side code

2. CORS:

   - Update `allow_origins` in `trpc_backend.py` with your frontend domain
   - Keep CORS restrictions as strict as possible

3. Monitoring:

   - Use `/trpc/health` endpoint for uptime monitoring
   - Monitor rate limit violations
   - Check logs for suspicious activity

4. Database:
   - Backup regularly
   - Monitor disk usage
   - Use prepared statements (already implemented)

## Error Handling

The API returns structured error responses:

```json
{
  "detail": "Error message",
  "timestamp": "2024-03-27T10:00:00Z"
}
```

## Health Check

Monitor system health:

```bash
curl https://your-api.onrender.com/trpc/health
```

## Development

1. Clone repository:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:

   ```bash
   export OPENAI_API_KEY=your_key
   export API_KEY=your_api_key
   ```

4. Run locally:
   ```bash
   uvicorn trpc_backend:app --reload
   ```

## License

MIT
