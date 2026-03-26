"""Spark Arena result upload to Firebase Storage."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from .auth import exchange_token, _user_agent

logger = logging.getLogger(__name__)


def generate_submission_id() -> str:
    """Generate a submission ID based on current timestamp."""
    return "sub%d" % int(time.time() * 1000)


def upload_file(
    id_token: str,
    bucket: str,
    user_id: str,
    submission_id: str,
    file_path: Path,
    folder: str,
) -> bool:
    """Upload a single file to Firebase Storage.

    Returns True on success, False on failure.
    """
    file_data = file_path.read_bytes()
    file_name = file_path.name

    object_name = "submissions/%s/%s/%s/%s" % (user_id, submission_id, folder, file_name)
    upload_url = "https://firebasestorage.googleapis.com/v0/b/%s/o?name=%s" % (
        bucket,
        quote(object_name, safe=""),
    )

    req = Request(upload_url, data=file_data, method="POST")
    req.add_header("Authorization", "Bearer %s" % id_token)
    req.add_header("Content-Type", "application/octet-stream")
    req.add_header("User-Agent", _user_agent())

    try:
        with urlopen(req, timeout=60) as resp:
            if resp.status == 200:
                logger.debug("Uploaded %s -> %s", file_name, object_name)
                return True
            else:
                logger.warning("Upload returned %d for %s", resp.status, file_name)
                return False
    except HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        logger.error("Upload failed (%d) for %s: %s", e.code, file_name, body)
        return False
    except URLError as e:
        logger.error("Upload error for %s: %s", file_name, e.reason)
        return False


def upload_benchmark_results(
    refresh_token: str,
    upload_files: list[tuple[Path, str]],
    submission_id: str | None = None,
) -> tuple[bool, str]:
    """Upload benchmark result files to Spark Arena.

    Args:
        refresh_token: User's refresh token.
        upload_files: List of ``(file_path, folder)`` tuples where *folder*
            is the Firebase Storage sub-folder (e.g. ``"recipes"``,
            ``"logs"``, ``"metadata"``).
        submission_id: Optional pre-generated submission ID.  One is created
            when not supplied.

    Returns:
        (success, submission_id) tuple.
    """
    # Exchange for fresh ID token
    result = exchange_token(refresh_token)
    id_token, user_id, bucket = result.id_token, result.user_id, result.bucket

    if submission_id is None:
        submission_id = generate_submission_id()

    all_ok = True
    for fpath, folder in upload_files:
        fpath = Path(fpath)
        if not fpath.is_file():
            logger.warning("Skipping missing file: %s", fpath)
            continue

        ok = upload_file(id_token, bucket, user_id, submission_id, fpath, folder)
        if not ok:
            all_ok = False

    return all_ok, submission_id
