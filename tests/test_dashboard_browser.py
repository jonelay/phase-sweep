"""Dashboard boot tests in a real browser, driven over CDP.

``test_server.py`` covers the REST and WebSocket surfaces through
``TestClient``, which never executes a line of JavaScript. Everything the
browser does -- resolving the ``/view-sweep/`` asset mount, importing the
ES module graph, mounting chrome and panels -- is unverified by it. The
regression this closes is real: view-sweep 0.3.0 changed
``/view-sweep/`` from one mount to three, and nothing in this repository
would have noticed the assets going missing.

The driver speaks the Chrome DevTools Protocol rather than using
``--dump-dom`` so that console errors and failed asset requests are
observable, not just the final DOM. Chromium is found on PATH or in the
Playwright cache; without one the module skips.

These tests assert that the dashboard *boots clean*, not that any panel
renders correct physics -- panels need job results, which belong to the
server tests.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import urllib.request
from importlib.util import find_spec
from pathlib import Path

import pytest

from tests.conftest import CHROMIUM_SKIP, SERVER_SKIP

if not find_spec("fastapi") or not find_spec("uvicorn"):
    pytest.skip(SERVER_SKIP, allow_module_level=True)

import uvicorn
from websockets.sync.client import connect as ws_connect

from phasesweep.server.app import ServerSettings, create_app
from tests.test_server import MOTOR_TOML

BOOT_SETTLE_S = 2.0
"""Quiet period after load, for module-graph imports and first fetches."""

TAB_IDS = ("results", "sweep", "validation", "editor")
"""app.js TABS; a tab that stops mounting is a boot regression."""


def find_chrome() -> str | None:
    """Chromium on PATH, else the newest Playwright download.

    Playwright has used both ``chrome-linux`` and ``chrome-linux64`` as
    the inner directory, so both are searched.
    """
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        if found := shutil.which(name):
            return found
    cache = Path.home() / ".cache" / "ms-playwright"
    downloads = sorted(cache.glob("chromium*/chrome-linux*/chrome"), reverse=True)
    return str(downloads[0]) if downloads else None


class CDP:
    """Minimal DevTools Protocol client: one page target, one socket."""

    def __init__(self, ws_url: str, timeout_s: float = 30.0) -> None:
        self._ws = ws_connect(ws_url, open_timeout=timeout_s, max_size=None)
        self._next_id = 0
        self.events: list[dict] = []

    def close(self) -> None:
        self._ws.close()

    def send(self, method: str, timeout_s: float = 30.0, **params) -> dict:
        """Call a method, buffering events that arrive before the reply."""
        self._next_id += 1
        msg_id = self._next_id
        self._ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no CDP reply to {method}")
            msg = json.loads(self._ws.recv(timeout=remaining))
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise AssertionError(f"{method} failed: {msg['error']}")
                return msg.get("result", {})
            if "method" in msg:
                self.events.append(msg)

    def drain(self, seconds: float) -> None:
        """Collect events for a quiet period; boot work outlives `load`."""
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                msg = json.loads(self._ws.recv(timeout=remaining))
            except TimeoutError:
                return
            if "method" in msg:
                self.events.append(msg)

    def wait_for(self, method: str, timeout_s: float) -> dict:
        for event in self.events:
            if event["method"] == method:
                return event
        deadline = time.monotonic() + timeout_s
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                msg = json.loads(self._ws.recv(timeout=remaining))
            except TimeoutError:
                break
            if "method" in msg:
                self.events.append(msg)
                if msg["method"] == method:
                    return msg
        raise TimeoutError(f"{method} not seen within {timeout_s}s")

    def evaluate(self, expression: str):
        """Evaluate JS in the page and return the JSON value."""
        result = self.send(
            "Runtime.evaluate", expression=expression, returnByValue=True,
            awaitPromise=True,
        )
        if result.get("exceptionDetails"):
            raise AssertionError(f"page threw: {result['exceptionDetails']}")
        return result["result"].get("value")


@pytest.fixture(scope="module")
def dashboard_url(tmp_path_factory):
    """The real FastAPI app on a real socket: a browser cannot use TestClient."""
    tmp = tmp_path_factory.mktemp("dashboard-browser")
    motors = tmp / "motors"
    motors.mkdir()
    (motors / "test_inrunner.toml").write_text(MOTOR_TOML)

    app = create_app(ServerSettings(
        motors_dir=motors, user_configs_dir=tmp / "user",
        output_dir=tmp / "output", workers=1, subtask_timeout_s=60.0,
    ))
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30.0
    while not server.started and time.monotonic() < deadline:
        if not thread.is_alive():
            pytest.fail("uvicorn thread died during startup")
        time.sleep(0.02)
    if not server.started:
        pytest.fail("uvicorn did not start within 30s")
    port = server.servers[0].sockets[0].getsockname()[1]

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=15)


@pytest.fixture(scope="module")
def booted(dashboard_url, tmp_path_factory):
    """Load the dashboard once; the three tests read the same boot."""
    chrome = find_chrome()
    if chrome is None:
        pytest.skip(CHROMIUM_SKIP)
    profile = tmp_path_factory.mktemp("chrome-profile")

    proc = subprocess.Popen(
        [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
         "--disable-dev-shm-usage", "--remote-debugging-port=0",
         f"--user-data-dir={profile}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    client = None
    try:
        port = _devtools_port(profile, proc)
        client = CDP(_page_target(port))
        for domain in ("Runtime", "Log", "Network", "Page"):
            client.send(f"{domain}.enable")
        client.send("Page.navigate", url=f"{dashboard_url}/")
        client.wait_for("Page.loadEventFired", timeout_s=30.0)
        client.drain(BOOT_SETTLE_S)
        yield client
    finally:
        if client is not None:
            client.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _devtools_port(profile: Path, proc: subprocess.Popen) -> int:
    """Chrome writes its chosen port to DevToolsActivePort once listening."""
    active = profile / "DevToolsActivePort"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"chrome exited with {proc.returncode} before listening")
        if active.is_file() and (text := active.read_text()).strip():
            return int(text.splitlines()[0])
        time.sleep(0.05)
    pytest.fail("chrome did not report a DevTools port within 30s")


def _page_target(port: int) -> str:
    """The about:blank page's WebSocket debugger URL."""
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/list", timeout=5) as r:
                targets = json.loads(r.read())
        except (OSError, json.JSONDecodeError):
            time.sleep(0.1)
            continue
        for target in targets:
            if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                return target["webSocketDebuggerUrl"]
        time.sleep(0.1)
    pytest.fail("no CDP page target within 30s")


# -- boot cleanliness ------------------------------------------------------------


def test_dashboard_boots_without_page_errors(booted):
    """No uncaught exception and no console error during boot.

    An ES module that fails to import is silent in the DOM and loud here.
    """
    thrown = [
        e["params"]["exceptionDetails"].get("text", "")
        + " "
        + str(e["params"]["exceptionDetails"].get("exception", {}).get("description", ""))
        for e in booted.events if e["method"] == "Runtime.exceptionThrown"
    ]
    assert not thrown, "uncaught exceptions during boot:\n" + "\n".join(thrown)

    console_errors = [
        "".join(str(a.get("value", a.get("description", ""))) for a in e["params"]["args"])
        for e in booted.events
        if e["method"] == "Runtime.consoleAPICalled" and e["params"]["type"] == "error"
    ]
    assert not console_errors, "console errors during boot:\n" + "\n".join(console_errors)

    assert booted.evaluate("document.readyState") == "complete"


def test_no_asset_request_failed(booted):
    """Every request the page made was served.

    This is the check that fails when the ``/view-sweep/`` mount shape
    changes underneath the dashboard.
    """
    failed = [
        (e["params"]["response"]["status"], e["params"]["response"]["url"])
        for e in booted.events
        if e["method"] == "Network.responseReceived"
        and e["params"]["response"]["status"] >= 400
    ]
    refused = [
        (e["params"].get("errorText"), e["params"].get("type"))
        for e in booted.events if e["method"] == "Network.loadingFailed"
    ]
    assert not failed, f"requests served with an error status: {failed}"
    assert not refused, f"requests that never completed: {refused}"


# -- view-sweep wiring -----------------------------------------------------------


def test_view_sweep_stylesheet_and_tokens_resolve(booted):
    """The mounted CSS actually applied, not merely 200'd.

    ``--vs-bg`` is defined by viewsweep.css; an empty value means the
    stylesheet was served but the cascade never saw it.
    """
    token = booted.evaluate(
        "getComputedStyle(document.documentElement)"
        ".getPropertyValue('--vs-bg').trim()"
    )
    assert token, "--vs-bg is empty: view-sweep chrome CSS did not apply"

    assert booted.evaluate("typeof window.Plotly") == "object", \
        "vendored Plotly did not load; plot panels cannot render"


# -- chrome and panels -----------------------------------------------------------


def test_chrome_and_all_tabs_mount(booted):
    """app.js reached the end of its mount path.

    Tabs are mounted late, after the module graph, the palette fetch and
    the first API calls, so their presence is a boot end-to-end signal.
    """
    assert booted.evaluate("!!document.querySelector('#topbar .vs-brand')")

    mounted = booted.evaluate(
        "Array.from(document.querySelectorAll('#main-tabs [role=tab]'))"
        ".map(t => t.id || t.dataset.tab || t.textContent.trim().toLowerCase())"
    )
    assert mounted, "no tabs mounted in #main-tabs"
    joined = " ".join(mounted).lower()
    missing = [t for t in TAB_IDS if t not in joined]
    assert not missing, f"tabs missing after boot: {missing} (mounted: {mounted})"

    assert booted.evaluate("document.querySelectorAll('#tab-panes > *').length") >= 1, \
        "tab panes host is empty"
