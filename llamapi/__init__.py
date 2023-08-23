import os

# Redis configurations
REDIS_BROKER_URL = os.environ.get(
    "LLAMAPI_REDIS_BROKER_URL", "redis://localhost:6379/0"
)
REDIS_BACKEND_URL = os.environ.get(
    "LLAMAPI_REDIS_BACKEND_URL", "redis://localhost:6379/1"
)

# Server side API configurations
API_HOST = os.environ.get("LLAMAPI_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("LLAMAPI_API_PORT", "8000"))

# Client side configurations
# This assumes that the client portforwarded the server port to localhost
HOST_URL = os.environ.get("LLAMAPI_HOST_URL", "http://localhost:8000")
