#!/usr/bin/env python3

import os
import platform
import json
from datetime import datetime


def check_environment():
    """Check the deployment environment and return details."""
    is_docker = os.path.exists(
        '/.dockerenv') or os.path.exists('/run/.containerenv')

    env_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "is_docker": is_docker,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "environment_vars": {
            "PORT": os.getenv("PORT"),
            "PYTHON_VERSION": os.getenv("PYTHON_VERSION"),
            "RENDER": os.getenv("RENDER"),
            "RENDER_SERVICE_ID": os.getenv("RENDER_SERVICE_ID")
        },
        "filesystem": {
            "dockerfile_exists": os.path.exists("/app/Dockerfile"),
            "requirements_exists": os.path.exists("/app/requirements.txt")
        }
    }
    return env_info


if __name__ == "__main__":
    info = check_environment()
    print("\nDeployment Environment Information:")
    print(json.dumps(info, indent=2))

    if info["is_docker"]:
        print("\n✅ Running in Docker container")
    else:
        print("\n❌ Not running in Docker container")
