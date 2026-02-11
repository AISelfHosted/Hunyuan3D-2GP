"""Version checking and update notification utility for Hunyuan3D-2GP.

Compares the local version (from setup.py) against the latest release tag
on the configured remote (GitHub / PyPI). Non-blocking and failure-tolerant
so it never prevents the app from starting.

Usage:
    from hy3dgen.version import check_for_updates
    update_info = check_for_updates()
    if update_info:
        print(f"Update available: {update_info['latest']} (you have {update_info['current']})")
"""

import logging
import re
from importlib.metadata import version as pkg_version, PackageNotFoundError

logger = logging.getLogger(__name__)

__version__ = "2.0.0"

# GitHub repo for checking latest release
GITHUB_REPO = "tencent/Hunyuan3D-2"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


def get_current_version() -> str:
    """Get the installed package version, falling back to __version__."""
    try:
        return pkg_version("hy3dgen")
    except PackageNotFoundError:
        return __version__


def _parse_version(v: str):
    """Parse version string into comparable tuple."""
    match = re.match(r'v?(\d+)\.(\d+)\.(\d+)', v)
    if match:
        return tuple(int(x) for x in match.groups())
    return (0, 0, 0)


def check_for_updates(timeout: float = 3.0) -> dict | None:
    """Check for a newer version on GitHub releases.

    Returns dict with 'current', 'latest', 'url' if update available,
    otherwise returns None. Never raises — failures are logged as warnings.
    """
    try:
        import urllib.request
        import json

        current = get_current_version()
        req = urllib.request.Request(
            GITHUB_API_URL,
            headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "Hunyuan3D-2GP"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())

        latest = data.get("tag_name", "")
        html_url = data.get("html_url", "")

        if _parse_version(latest) > _parse_version(current):
            logger.info(f"Update available: {latest} (current: {current})")
            return {"current": current, "latest": latest, "url": html_url}

        logger.info(f"Version {current} is up to date")
        return None

    except Exception as e:
        logger.warning(f"Update check failed (non-blocking): {e}")
        return None
