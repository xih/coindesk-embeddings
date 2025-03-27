#!/usr/bin/env python3

import secrets
import string
import argparse
import re
import httpx
import os
from datetime import datetime


def generate_api_key(prefix="sk", length=48):
    """Generate a secure API key with given prefix and length."""
    alphabet = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}_{random_string}"


def validate_api_key_format(api_key):
    """Validate the format of the API key."""
    pattern = r'^sk_[A-Za-z0-9]{48}$'
    return bool(re.match(pattern, api_key))


def test_api_key(api_key, base_url="http://localhost:8000"):
    """Test the API key against the endpoints."""
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    # Test health endpoint
    try:
        health_response = httpx.get(f"{base_url}/trpc/health", headers=headers)
        print("\nHealth Check Response:", health_response.status_code)
        print(health_response.json())
    except Exception as e:
        print(f"Error testing health endpoint: {str(e)}")

    # Test query endpoint
    try:
        query_data = {
            "query": "What is Bitcoin?",
            "max_tokens": 100
        }
        query_response = httpx.post(
            f"{base_url}/trpc/query", headers=headers, json=query_data)
        print("\nQuery Test Response:", query_response.status_code)
        print(query_response.json())
    except Exception as e:
        print(f"Error testing query endpoint: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate and test API keys for the RAG API')
    parser.add_argument('--test', action='store_true',
                        help='Test the generated API key')
    parser.add_argument(
        '--url', default="http://localhost:8000", help='Base URL for testing')
    args = parser.parse_args()

    # Generate new API key
    api_key = generate_api_key()
    print(f"\nGenerated API Key: {api_key}")

    # Validate format
    if validate_api_key_format(api_key):
        print("✅ API key format is valid")
    else:
        print("❌ API key format is invalid")

    # Save to .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Read existing .env file
        env_contents = ""
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                env_contents = f.read()

        # Update or add API_KEY
        if 'API_KEY=' in env_contents:
            env_contents = re.sub(
                r'API_KEY=.*\n', f'API_KEY={api_key}\n', env_contents)
        else:
            env_contents += f'\n# Generated on {timestamp}\nAPI_KEY={api_key}\n'

        # Write back to .env file
        with open(env_path, 'w') as f:
            f.write(env_contents.strip() + '\n')
        print(f"✅ API key saved to {env_path}")

    except Exception as e:
        print(f"❌ Error saving to .env file: {str(e)}")
        print(
            f"Please manually set this environment variable: API_KEY={api_key}")

    # Test the API key if requested
    if args.test:
        print(f"\nTesting API key against {args.url}...")
        test_api_key(api_key, args.url)
        print("\nRemember to set this API key in your Render environment variables!")


if __name__ == "__main__":
    main()
