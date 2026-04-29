"""Spark Arena authentication via auth.sparkrun.dev proxy."""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from dataclasses import dataclass
import contextlib

logger = logging.getLogger(__name__)

AUTH_PROXY_BASE = "https://auth.sparkrun.dev"
_CALLBACK_PORT = 9005


@dataclass
class ExchangeResult:
    """Result from a token exchange with the auth proxy."""

    id_token: str
    user_id: str
    bucket: str
    email: str | None = None
    display_name: str | None = None
    provider: str | None = None


def _user_agent() -> str:
    """Return User-Agent string for auth proxy requests."""
    from sparkrun import __version__

    return "sparkrun/%s" % __version__


def get_token_path() -> Path:
    """Return path to the persisted refresh token file."""
    return Path.home() / ".config" / "sparkrun" / "arena_token"


def save_refresh_token(token: str) -> None:
    """Persist refresh token to disk with restricted permissions."""
    path = get_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token)
    os.chmod(path, 0o600)


def load_refresh_token() -> str | None:
    """Load refresh token from disk, or None if not present."""
    path = get_token_path()
    if not path.is_file():
        return None
    return path.read_text().strip() or None


def clear_refresh_token() -> None:
    """Delete the stored refresh token."""
    path = get_token_path()
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def exchange_token(refresh_token: str, debug_mode: bool = False) -> ExchangeResult:
    """Exchange a refresh token for user credentials via auth proxy.

    Raises ``RuntimeError`` on failure.
    """
    url = "%s/exchange" % AUTH_PROXY_BASE
    payload = json.dumps({"refresh_token": refresh_token}).encode()
    req = Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": _user_agent(),
        },
    )
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            err_data = json.loads(body)
            msg = err_data.get("message") or err_data.get("error", body)
        except (json.JSONDecodeError, ValueError):
            msg = body
        raise RuntimeError("Token exchange failed (%d): %s" % (e.code, msg))
    except URLError as e:
        raise RuntimeError("Cannot reach auth proxy: %s" % e.reason)

    id_token = data.get("id_token")
    user_id = data.get("user_id")
    bucket = data.get("bucket")
    if not id_token or not user_id or not bucket:
        raise RuntimeError("Incomplete exchange response")
    if debug_mode:  # only show exchange data under explicit debug (not just all debug logging)
        logger.debug("Exchange data: %s", data)
    return ExchangeResult(
        id_token=id_token,
        user_id=user_id,
        bucket=bucket,
        email=data.get("email"),
        display_name=data.get("display_name"),
        provider=data.get("provider"),
    )


def is_logged_in() -> bool:
    """Check whether a valid token exists and can be exchanged."""
    token = load_refresh_token()
    if not token:
        return False
    try:
        exchange_token(token)
        return True
    except RuntimeError:
        return False


def generate_challenge_id() -> str:
    """Generate a human-readable challenge code (XXXXX-XXXXX)."""
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    chars = [secrets.choice(charset) for _ in range(10)]
    return "".join(chars[:5]) + "-" + "".join(chars[5:])


def _can_open_browser() -> bool:
    """Check if we're in an environment that can open a browser."""
    if not sys.stdin.isatty():
        return False
    if os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"):
        return False
    if sys.platform == "linux" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    return True


def run_browser_login() -> str | None:
    """Run browser-based OAuth login flow.

    Returns the refresh token on success, or None on failure.
    """
    challenge_id = generate_challenge_id()
    received_token: list[str] = []
    server_error: list[str] = []

    class CallbackHandler(BaseHTTPRequestHandler):
        def _send_cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", AUTH_PROXY_BASE)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length else b""
            try:
                data = json.loads(body)
                token = data.get("token")
            except (json.JSONDecodeError, ValueError):
                token = None

            if token:
                received_token.append(token)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            else:
                server_error.append("no token in POST body")
                self.send_response(400)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error":"missing token"}')

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return

            params = parse_qs(parsed.query)
            token_list = params.get("token", [])
            if token_list:
                received_token.append(token_list[0])
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authentication received.</h2><p>You can close this tab and return to the terminal.</p></body></html>"
                )
            else:
                error_list = params.get("error", ["unknown error"])
                server_error.append(error_list[0])
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Authentication failed.")

        def log_message(self, format, *args):
            logger.debug("callback server: %s", format % args)

    from sparkrun.orchestration.primitives import find_available_port

    callback_port = find_available_port("localhost", _CALLBACK_PORT)

    # noinspection PyTypeChecker
    server = HTTPServer(("127.0.0.1", callback_port), CallbackHandler)
    server.timeout = 300  # 5 minutes

    callback_url = "http://localhost:%d/callback" % callback_port
    login_url = "%s/login?%s" % (
        AUTH_PROXY_BASE,
        urlencode({"callback": callback_url, "code": challenge_id}),
    )

    import webbrowser

    if not webbrowser.open(login_url):
        return None

    # Wait for the token via POST from browser JS
    while not received_token and not server_error:
        server.handle_request()

    server.server_close()

    if server_error:
        logger.error("Browser auth error: %s", server_error[0])
        return None

    if not received_token:
        return None

    return received_token[0], challenge_id


def run_device_code_login() -> str | None:
    """Run device-code login flow for headless environments.

    Returns the refresh token on success, or None on failure.
    """
    import time

    # Request a device code
    url = "%s/device/code" % AUTH_PROXY_BASE
    req = Request(
        url,
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "User-Agent": _user_agent(),
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (URLError, HTTPError) as e:
        logger.error("Failed to request device code: %s", e)
        return None

    device_code = data.get("device_code")
    user_code = data.get("user_code")
    verification_url = data.get("verification_url", "%s/device" % AUTH_PROXY_BASE)
    interval = data.get("interval", 5)
    expires_in = data.get("expires_in", 900)

    if not device_code or not user_code:
        logger.error("Incomplete device code response")
        return None

    import click

    click.echo()
    click.echo("To sign in, open this URL on any device:")
    click.echo("  %s" % verification_url)
    click.echo()
    click.echo("and enter code: %s" % user_code)
    click.echo()
    click.echo("Waiting for authentication... (press Ctrl+C to cancel)")

    # Poll for completion
    poll_url = "%s/device/token" % AUTH_PROXY_BASE
    deadline = time.monotonic() + expires_in

    while time.monotonic() < deadline:
        time.sleep(interval)

        payload = json.dumps({"device_code": device_code}).encode()
        poll_req = Request(
            poll_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": _user_agent(),
            },
            method="POST",
        )
        try:
            with urlopen(poll_req, timeout=15) as resp:
                poll_data = json.loads(resp.read())
        except HTTPError as e:
            if e.code == 202:
                # Still pending
                continue
            if e.code == 410:
                click.echo("Device code expired. Please try again.")
                return None
            logger.debug("Poll error: %d", e.code)
            continue
        except URLError:
            continue

        refresh_token = poll_data.get("refresh_token")
        if refresh_token:
            return refresh_token

        if poll_data.get("status") == "pending":
            continue

    click.echo("Timed out waiting for authentication.")
    return None


def run_login_flow(force_browser: bool = False, force_device: bool = False) -> bool:
    """Run the appropriate login flow. Returns True on success."""
    import click

    if force_browser:
        use_browser = True
    elif force_device:
        use_browser = False
    else:
        use_browser = _can_open_browser()

    if use_browser:
        click.echo("Opening browser for authentication...")
        result = run_browser_login()
        if result is None:
            if not force_browser:
                click.echo("Browser not available. Falling back to device code flow.")
                return _complete_device_login()
            click.echo("Browser login failed.", err=True)
            return False

        refresh_token, challenge_id = result

        # Prompt for challenge verification
        entered = click.prompt("Enter the code displayed in the browser")
        if entered.strip().upper() != challenge_id:
            click.echo("Incorrect code. Login failed.", err=True)
            return False

        return _complete_login(refresh_token)
    else:
        if not force_device:
            click.echo("Browser not available. Using device code flow.")
        return _complete_device_login()


def _complete_device_login() -> bool:
    """Run device code flow and complete login."""
    refresh_token = run_device_code_login()
    if not refresh_token:
        return False
    return _complete_login(refresh_token)


def _complete_login(refresh_token: str) -> bool:
    """Verify token, persist, and report success."""
    import click

    logger.debug("Token to exchange: len=%d", len(refresh_token))
    try:
        result = exchange_token(refresh_token)
    except RuntimeError as e:
        click.echo("Login verification failed: %s" % e, err=True)
        return False

    save_refresh_token(refresh_token)
    if result.email:
        click.echo("Logged in as %s." % result.email)
    else:
        click.echo("Logged in successfully.")
    return True
