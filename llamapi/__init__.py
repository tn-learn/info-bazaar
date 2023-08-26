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

# Worker side configurations
LLAMAPI_GLOBAL_HF_CACHE_DIRECTORY = os.environ.get(
    "LLAMAPI_GLOBAL_HF_CACHE_DIRECTORY", "/tmp/huggingface_cache"
)
LLAMAPI_LOCAL_HF_CACHE_DIRECTORY = os.environ.get(
    "LLAMAPI_LOCAL_HF_CACHE_DIRECTORY", "/tmp/huggingface_cache"
)


# Client side configurations
# This assumes that the client portforwarded the server port to localhost
HOST_URL = os.environ.get("LLAMAPI_HOST_URL", "http://localhost:8000")
POLL_INTERVAL = int(os.environ.get("LLAMAPI_POLL_INTERVAL", 2))
