
import base64
import json
import logging
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone

import requests


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

API_ORIGIN = "http://telecom-churn-api:8000"

GITHUB_OWNER = "DSAIKIRANREDDY"
GITHUB_REPO = "telecom-churn-config"
GITHUB_BRANCH = "main"
GITHUB_FILE_PATH = "config.json"

GITHUB_TOKEN = os.getenv("GITHUB_CONFIG_WRITE_TOKEN")

GITHUB_API_URL = (
    f"https://api.github.com/repos/"
    f"{GITHUB_OWNER}/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
)

TUNNEL_URL_PATTERN = re.compile(
    r"https://[a-zA-Z0-9-]+\.trycloudflare\.com"
)

RESTART_DELAY = 10
HEALTH_CHECK_RETRIES = 30
HEALTH_CHECK_INTERVAL = 2


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Shutdown handling
# ---------------------------------------------------------

shutdown_requested = False


def handle_shutdown(signum, frame):
    global shutdown_requested

    logger.info("Shutdown signal received.")
    shutdown_requested = True


signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


# ---------------------------------------------------------
# GitHub configuration publisher
# ---------------------------------------------------------

def update_github_config(api_base_url: str, status: str):
    """
    Update the private GitHub config.json file.

    The GitHub Contents API requires the current file SHA
    when updating an existing file.
    """

    if not GITHUB_TOKEN:
        raise RuntimeError(
            "GITHUB_CONFIG_WRITE_TOKEN is not configured."
        )

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "telecom-churn-tunnel-manager",
    }

    # Get the existing file and its SHA.
    response = requests.get(
        GITHUB_API_URL,
        headers=headers,
        params={"ref": GITHUB_BRANCH},
        timeout=15,
    )

    response.raise_for_status()

    file_data = response.json()
    file_sha = file_data["sha"]

    # Build updated configuration.
    config = {
        "api_base_url": api_base_url,
        "status": status,
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    content = json.dumps(
        config,
        indent=4,
    )

    encoded_content = base64.b64encode(
        content.encode("utf-8")
    ).decode("utf-8")

    # Update the file on GitHub.
    payload = {
        "message": f"Update API tunnel status: {status}",
        "content": encoded_content,
        "sha": file_sha,
        "branch": GITHUB_BRANCH,
    }

    response = requests.put(
        GITHUB_API_URL,
        headers=headers,
        json=payload,
        timeout=15,
    )

    response.raise_for_status()

    logger.info(
        "GitHub configuration updated. Status: %s",
        status,
    )


# ---------------------------------------------------------
# Public tunnel health verification
# ---------------------------------------------------------

def verify_tunnel(api_base_url: str) -> bool:
    """
    Check whether the public tunnel reaches FastAPI.
    """

    health_url = f"{api_base_url}/health/live"

    for attempt in range(1, HEALTH_CHECK_RETRIES + 1):

        if shutdown_requested:
            return False

        try:
            response = requests.get(
                health_url,
                timeout=5,
            )

            if response.ok:
                logger.info(
                    "Public tunnel health check successful."
                )
                return True

        except requests.RequestException:
            pass

        logger.info(
            "Waiting for tunnel health check (%s/%s)...",
            attempt,
            HEALTH_CHECK_RETRIES,
        )

        time.sleep(HEALTH_CHECK_INTERVAL)

    logger.error("Public tunnel health check failed.")

    return False


# ---------------------------------------------------------
# Cloudflare tunnel process
# ---------------------------------------------------------

def run_tunnel():

    command = [
        "cloudflared",
        "tunnel",
        "--url",
        API_ORIGIN,
        "--output",
        "json",
    ]

    logger.info("Starting Cloudflare Quick Tunnel...")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tunnel_url = None
    tunnel_published = False

    try:

        for line in process.stdout:

            if shutdown_requested:
                break

            # Print Cloudflare logs without exposing secrets.
            logger.info("cloudflared: %s", line.rstrip())

            # Detect the generated Quick Tunnel URL.
            if tunnel_url is None:

                match = TUNNEL_URL_PATTERN.search(line)

                if match:
                    tunnel_url = match.group(0)

                    logger.info(
                        "Quick Tunnel URL detected: %s",
                        tunnel_url,
                    )

                    # Verify the URL before publishing it.
                    if verify_tunnel(tunnel_url):

                        update_github_config(
                            api_base_url=tunnel_url,
                            status="active",
                        )

                        tunnel_published = True

                    else:
                        logger.error(
                            "Tunnel URL was not published "
                            "because health verification failed."
                        )

        return_code = process.wait()

        logger.warning(
            "cloudflared exited with code %s.",
            return_code,
        )

    finally:

        if process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        if process.stdout:
            process.stdout.close()

    return tunnel_published


# ---------------------------------------------------------
# Main supervisor
# ---------------------------------------------------------

def main():

    if not GITHUB_TOKEN:
        raise RuntimeError(
            "Missing GITHUB_CONFIG_WRITE_TOKEN environment variable."
        )

    logger.info("Cloudflare Manager started.")

    while not shutdown_requested:

        try:
            run_tunnel()

        except Exception:
            logger.exception(
                "An error occurred while running the tunnel."
            )

        if shutdown_requested:
            break

        # Mark the old URL inactive before restarting.
        try:
            update_github_config(
                api_base_url="",
                status="inactive",
            )
        except Exception:
            logger.exception(
                "Could not mark GitHub configuration inactive."
            )

        logger.info(
            "Restarting Cloudflare Tunnel in %s seconds...",
            RESTART_DELAY,
        )

        # Wait without delaying shutdown.
        for _ in range(RESTART_DELAY):
            if shutdown_requested:
                break
            time.sleep(1)

    # Final cleanup.
    try:
        update_github_config(
            api_base_url="",
            status="inactive",
        )
    except Exception:
        logger.exception(
            "Could not publish final inactive status."
        )

    logger.info("Cloudflare Manager stopped.")


if __name__ == "__main__":
    main()
