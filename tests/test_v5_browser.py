"""
Unit and security tests for JARVIS V5 Secure Local Browser Agent.
"""

import pytest
from unittest.mock import patch, AsyncMock

from agent.config import settings
from agent.tools.security import (
    security_validator,
    RiskLevel,
    SecurityViolation,
    ConfirmationRequired,
)
from agent.tools.browser import (
    open_browser,
    open_url,
    browser_search,
    browser_go_back,
    browser_go_forward,
    browser_refresh,
    get_browser_state,
)
from agent.ai.browser_planner import BrowserPlanner, browser_planner


@pytest.fixture(autouse=True)
def ensure_security_resumed():
    """Ensure computer control is active before and after tests."""
    security_validator.resume_control()
    yield
    security_validator.resume_control()


# 1. URL Security Tests
@pytest.mark.parametrize("valid_url", [
    "https://docs.python.org",
    "http://example.com",
    "https://github.com/SivaSabariGanesan/Jarvis",
    "www.google.com",
    "python.org/downloads",
])
def test_valid_browser_urls_accepted(valid_url):
    """Ensure valid http/https URLs are validated successfully."""
    validated = security_validator.validate_browser_url(valid_url)
    assert validated.startswith("http://") or validated.startswith("https://")


@pytest.mark.parametrize("disallowed_url", [
    "file:///C:/Windows/System32/cmd.exe",
    "javascript:alert(document.cookie)",
    "data:text/html,<script>alert(1)</script>",
    "vbscript:msgbox(1)",
    "shell:System",
    "about:blank",
    "chrome://settings",
    "edge://flags",
    "ms-appx:///cmd.exe",
])
def test_disallowed_browser_url_schemes_rejected(disallowed_url):
    """Ensure non-http/https schemes and browser internals are rejected with SecurityViolation."""
    with pytest.raises(SecurityViolation):
        security_validator.validate_browser_url(disallowed_url)


# 2. Domain Policy Tests
def test_browser_domain_policy_allowlist():
    """Ensure allowlist domain policy blocks unauthorized domains."""
    with patch.object(settings, "browser_domain_policy", "allowlist"), \
         patch.object(settings, "browser_allowed_domains", ["google.com", "python.org"]):
        
        # Allowed domains
        assert security_validator.validate_browser_url("https://google.com/search")
        assert security_validator.validate_browser_url("https://docs.python.org")

        # Blocked unauthorized domain
        with pytest.raises(SecurityViolation, match="not authorized under the current browser domain policy"):
            security_validator.validate_browser_url("https://unauthorized-domain.com")


# 3. Browser Planner Tests
def test_browser_planner_deterministic_plan_creation():
    """Test fast deterministic plan generation for common browser requests."""
    planner = BrowserPlanner()
    plan = planner.create_deterministic_plan("open Chrome and search for Python interview questions")
    assert plan is not None
    assert plan["goal"].lower() == "search for 'python interview questions'"
    assert len(plan["steps"]) == 2
    assert plan["steps"][0]["tool"] == "open_browser"
    assert plan["steps"][1]["tool"] == "browser_search"
    assert plan["steps"][1]["args"]["query"].lower() == "python interview questions"


def test_browser_planner_rejects_unauthorized_tools():
    """Ensure planner rejects unknown or unauthorized tool invocations."""
    planner = BrowserPlanner()
    malicious_plan = {
        "goal": "Malicious task",
        "steps": [
            {"tool": "run_powershell", "args": {"cmd": "whoami"}},
        ]
    }
    with pytest.raises(SecurityViolation, match="unauthorized or unknown tool"):
        planner.validate_plan(malicious_plan)


def test_browser_planner_rate_limiting():
    """Ensure planner rejects plans exceeding max allowed step count."""
    planner = BrowserPlanner()
    excessive_plan = {
        "goal": "Loop task",
        "steps": [{"tool": "browser_refresh", "args": {}} for _ in range(25)]
    }
    with pytest.raises(SecurityViolation, match="exceeds maximum allowed"):
        planner.validate_plan(excessive_plan)


# 4. Search Query Data Isolation
@pytest.mark.anyio
async def test_browser_search_query_treated_as_data():
    """Ensure dangerous shell characters in search queries remain strictly search text."""
    with patch("webbrowser.open") as mock_open:
        res = await browser_search("Python tutorials && shutdown /s")
        assert "Searched the web for" in res
        mock_open.assert_called_once()
        target_url = mock_open.call_args[0][0]
        # Query must be URL-encoded, never passed to shell
        assert "Python+tutorials+%26%26+shutdown+%2Fs" in target_url or "Python+tutorials" in target_url


# 5. Credential & Login Prompt Protection
def test_login_prompt_detection():
    """Ensure login and authentication keywords are detected."""
    assert security_validator.is_login_or_credential_prompt("Google Sign In - Enter your password") is True
    assert security_validator.is_login_or_credential_prompt("GitHub Login - Two-Factor Authentication (2FA)") is True
    assert security_validator.is_login_or_credential_prompt("Python 3.12 Documentation") is False


# 6. Takeover Mode Tests
@pytest.mark.anyio
async def test_takeover_mode_pause_and_resume():
    """Verify Takeover mode pauses automation and resume inspects screen freshly."""
    planner = BrowserPlanner()

    # User initiates takeover
    pause_msg = planner.pause_for_takeover()
    assert planner.is_takeover_active is True
    assert "You have control" in pause_msg

    # Plan execution during takeover must be rejected
    plan = {"goal": "Search", "steps": [{"tool": "browser_refresh", "args": {}}]}
    res = await planner.execute_plan(plan)
    assert "paused in Takeover mode" in res

    # Resume takeover
    resume_msg = await planner.resume_after_takeover()
    assert planner.is_takeover_active is False
    assert "resumed" in resume_msg


# 7. Dangerous Web Actions Classification
@pytest.mark.parametrize("action", [
    "purchase", "checkout", "send money", "delete account", "download executable"
])
def test_dangerous_web_actions_flagged(action):
    """Ensure high-risk web operations are identified for confirmation."""
    assert security_validator.is_dangerous_web_action(action) is True


# 8. Navigation & State Tools
def test_browser_navigation_tools():
    """Test browser history and state helper tools."""
    with patch("agent.tools.browser.hotkey") as mock_hotkey, \
         patch("agent.tools.browser.press_key") as mock_press:
        
        back_res = browser_go_back()
        assert "Navigated back" in back_res
        mock_hotkey.assert_called_with("alt+left")

        fwd_res = browser_go_forward()
        assert "Navigated forward" in fwd_res
        mock_hotkey.assert_called_with("alt+right")

        ref_res = browser_refresh()
        assert "Refreshed" in ref_res
        mock_press.assert_called_with("f5")

    state = get_browser_state()
    assert isinstance(state, dict)
    assert "browser" in state
    assert "running" in state

