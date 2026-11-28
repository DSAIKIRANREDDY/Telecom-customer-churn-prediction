
import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader


# Extract API key from the HTTP request header
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(
    api_key: str | None = Security(api_key_header)
) -> str:

    expected_api_key = os.getenv("API_KEY")

    # Fail closed if the backend has no configured API key
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API authentication is not configured"
        )

    # Reject missing or incorrect API keys
    if (
        api_key is None
        or not secrets.compare_digest(api_key, expected_api_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )

    return api_key
