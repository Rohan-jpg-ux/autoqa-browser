"""
Browser automation tool for AutoQA.
Wraps Playwright with safe, audited actions.
Every action is logged and screenshots are taken at each step.
"""

import os
import time
import base64
import tempfile
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Safety limits
MAX_ACTIONS_PER_TEST = 30
ALLOWED_SCHEMES = ("http://", "https://")
BLOCKED_DOMAINS = [
    "localhost", "127.0.0.1", "0.0.0.0",
    "169.254.", "10.", "192.168.", "172.16.",
]


def is_url_safe(url: str) -> bool:
    """Only allow public HTTP/HTTPS URLs — block local network access"""
    url = url.lower().strip()
    if not any(url.startswith(s) for s in ALLOWED_SCHEMES):
        return False
    for blocked in BLOCKED_DOMAINS:
        if blocked in url:
            return False
    return True


class BrowserSession:
    """
    Manages a single Playwright browser session.
    All actions are logged. Screenshots taken at each step.
    """

    def __init__(self, headless: bool = True, timeout: int = 10000):
        self.headless = headless
        self.timeout = timeout
        self.browser = None
        self.page = None
        self.context = None
        self.action_log = []
        self.screenshots = []
        self.action_count = 0
        self._playwright = None

    def start(self):
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self.browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        self.context = self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="AutoQA-Bot/1.0 (automated testing)",
        )
        self.page = self.context.new_page()
        self.page.set_default_timeout(self.timeout)
        logger.info("Browser session started")

    def stop(self):
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception as e:
            logger.warning(f"Error stopping browser: {e}")
        logger.info("Browser session stopped")

    def _check_limit(self):
        if self.action_count >= MAX_ACTIONS_PER_TEST:
            raise RuntimeError(f"Safety limit: max {MAX_ACTIONS_PER_TEST} actions per test reached")
        self.action_count += 1

    def _log(self, action: str, detail: str, success: bool, error: str = None):
        entry = {
            "step": len(self.action_log) + 1,
            "action": action,
            "detail": detail,
            "success": success,
            "error": error,
            "timestamp": time.time(),
        }
        self.action_log.append(entry)
        status = "✅" if success else "❌"
        logger.info(f"{status} [{action}] {detail}")
        return entry

    def screenshot(self, label: str = "") -> str:
        """Take a screenshot and return base64 encoded string"""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            self.page.screenshot(path=tmp.name, full_page=False)
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            self.screenshots.append({
                "label": label or f"step_{len(self.screenshots)+1}",
                "path": tmp.name,
                "b64": b64,
            })
            return tmp.name
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")
            return None

    # ── Safe Actions ──────────────────────────────────────────────────────────

    def navigate(self, url: str) -> dict:
        self._check_limit()
        if not is_url_safe(url):
            return self._log("navigate", url, False, "URL blocked by safety policy")
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
            self.screenshot(f"navigate_{url[:40]}")
            return self._log("navigate", url, True)
        except Exception as e:
            self.screenshot("navigate_error")
            return self._log("navigate", url, False, str(e))

    def click(self, selector: str) -> dict:
        self._check_limit()
        try:
            self.page.click(selector, timeout=8000)
            time.sleep(0.5)
            self.screenshot(f"click_{selector[:30]}")
            return self._log("click", selector, True)
        except Exception as e:
            self.screenshot("click_error")
            return self._log("click", selector, False, str(e))

    def click_text(self, text: str) -> dict:
        self._check_limit()
        try:
            self.page.get_by_text(text, exact=False).first.click(timeout=8000)
            time.sleep(0.5)
            self.screenshot(f"click_text_{text[:30]}")
            return self._log("click_text", text, True)
        except Exception as e:
            self.screenshot("click_text_error")
            return self._log("click_text", text, False, str(e))

    def type_text(self, selector: str, text: str) -> dict:
        self._check_limit()
        # Mask passwords in logs
        display_text = "***" if any(k in selector.lower() for k in ["password", "secret", "token"]) else text
        try:
            self.page.fill(selector, text, timeout=8000)
            self.screenshot(f"type_{selector[:20]}")
            return self._log("type", f"{selector} → '{display_text}'", True)
        except Exception as e:
            self.screenshot("type_error")
            return self._log("type", f"{selector} → '{display_text}'", False, str(e))

    def press_key(self, key: str) -> dict:
        self._check_limit()
        try:
            self.page.keyboard.press(key)
            time.sleep(0.3)
            self.screenshot(f"key_{key}")
            return self._log("press_key", key, True)
        except Exception as e:
            return self._log("press_key", key, False, str(e))

    def wait_for_text(self, text: str, timeout: int = 8000) -> dict:
        self._check_limit()
        try:
            self.page.wait_for_selector(f"text={text}", timeout=timeout)
            self.screenshot(f"wait_text_{text[:20]}")
            return self._log("wait_for_text", text, True)
        except Exception as e:
            self.screenshot("wait_text_error")
            return self._log("wait_for_text", text, False, str(e))

    def wait_for_selector(self, selector: str, timeout: int = 8000) -> dict:
        self._check_limit()
        try:
            self.page.wait_for_selector(selector, timeout=timeout)
            self.screenshot(f"wait_sel_{selector[:20]}")
            return self._log("wait_for_selector", selector, True)
        except Exception as e:
            self.screenshot("wait_sel_error")
            return self._log("wait_for_selector", selector, False, str(e))

    def assert_text_visible(self, text: str) -> dict:
        self._check_limit()
        try:
            element = self.page.get_by_text(text, exact=False).first
            is_visible = element.is_visible()
            self.screenshot(f"assert_{text[:20]}")
            return self._log("assert_text_visible", f"'{text}' visible={is_visible}", is_visible,
                             None if is_visible else f"Text '{text}' not visible on page")
        except Exception as e:
            return self._log("assert_text_visible", text, False, str(e))

    def assert_url_contains(self, substring: str) -> dict:
        self._check_limit()
        current = self.page.url
        success = substring.lower() in current.lower()
        return self._log("assert_url_contains", f"'{substring}' in '{current}'", success,
                         None if success else f"URL '{current}' doesn't contain '{substring}'")

    def assert_title_contains(self, text: str) -> dict:
        self._check_limit()
        title = self.page.title()
        success = text.lower() in title.lower()
        return self._log("assert_title_contains", f"'{text}' in title='{title}'", success,
                         None if success else f"Title '{title}' doesn't contain '{text}'")

    def scroll_down(self, pixels: int = 500) -> dict:
        self._check_limit()
        try:
            self.page.evaluate(f"window.scrollBy(0, {pixels})")
            time.sleep(0.3)
            self.screenshot("scroll")
            return self._log("scroll_down", f"{pixels}px", True)
        except Exception as e:
            return self._log("scroll_down", f"{pixels}px", False, str(e))

    def get_page_text(self) -> str:
        try:
            return self.page.inner_text("body")[:5000]
        except Exception:
            return ""

    def get_current_url(self) -> str:
        try:
            return self.page.url
        except Exception:
            return ""
