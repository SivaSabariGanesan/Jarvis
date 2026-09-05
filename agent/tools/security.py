"""
Security validation and audit logging layer for JARVIS V3.
Enforces a deny-by-default architecture, risk-level checks, path sandboxing,
URL safety, explicit confirmation for high-risk tools, and emergency stop.
"""

import os
import re
import enum
import logging
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from agent.config import settings

logger = logging.getLogger("jarvis.security")


class RiskLevel(str, enum.Enum):
    """Risk tiers for computer control actions."""
    LOW = "LOW"        # Read-only or safe standard operations (system info, screenshot, open allowlisted apps)
    MEDIUM = "MEDIUM"  # Non-destructive state changes within sandbox (create file/folder, rename, volume)
    HIGH = "HIGH"      # Destructive or high-impact actions (delete file, shutdown, kill process)


class ControlState(str, enum.Enum):
    """Operational status of the computer control layer."""
    ENABLED = "ENABLED"
    PAUSED = "PAUSED"


class SecurityViolation(Exception):
    """Raised when an operation violates security constraints."""
    pass


class ConfirmationRequired(Exception):
    """Raised when a high-risk operation requires explicit user confirmation."""
    def __init__(self, message: str, tool_name: str, arguments: Dict[str, Any]):
        super().__init__(message)
        self.tool_name = tool_name
        self.arguments = arguments


# Positive confirmation tokens (strict matching)
POSITIVE_CONFIRMATION_TOKENS: Set[str] = {
    "yes",
    "confirm",
    "do it",
    "go ahead",
    "proceed",
    "yes please",
    "yes, proceed",
    "affirmative",
    "authorized",
    "execute",
}

# Ambiguous or negative tokens that MUST NOT count as confirmation
REJECTED_CONFIRMATION_TOKENS: Set[str] = {
    "maybe",
    "okay",
    "ok",
    "sure",
    "sure?",
    "i guess",
    "perhaps",
    "no",
    "cancel",
    "stop",
    "abort",
    "nevermind",
}

# V4 Keyboard & Mouse Security Controls
ALLOWED_KEYS: Set[str] = {
    # Alphanumeric & Symbols
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "space", "tab", "enter", "return", "backspace", "delete", "del", "esc", "escape",
    # Navigation
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown", "insert",
    # Modifiers
    "ctrl", "control", "alt", "shift", "win", "windows",
    # Function keys
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
}

# Explicit Hotkey Allowlists
SAFE_HOTKEYS: Set[str] = {
    "ctrl+c", "ctrl+v", "ctrl+x", "ctrl+a", "ctrl+z", "ctrl+y", "ctrl+s", "ctrl+f",
    "ctrl+p", "ctrl+w", "ctrl+t", "ctrl+r", "ctrl+n", "ctrl+o", "ctrl+l", "ctrl+k",
    "alt+tab", "alt+left", "alt+right", "alt+enter",
    "win+d", "win+e", "win+r", "win+s", "win+l", "win+tab", "win+up", "win+down", "win+left", "win+right",
    "ctrl+shift+esc", "ctrl+shift+t", "ctrl+shift+n",
}

DANGEROUS_HOTKEYS: Set[str] = {
    "shift+delete",
    "alt+f4",
    "ctrl+alt+delete",
    "ctrl+alt+del",
}

# Dangerous / Sensitive UI labels requiring explicit user authorization
PROTECTED_UI_ELEMENTS: Set[str] = {
    "delete",
    "permanently delete",
    "format",
    "uninstall",
    "reset",
    "shutdown",
    "shut down",
    "restart",
    "factory reset",
    "remove account",
    "disable security",
    "erase",
    "wipe",
}



@dataclass
class ValidationResult:
    is_valid: bool
    risk_level: RiskLevel
    sanitized_args: Dict[str, Any]
    error_message: Optional[str] = None
    requires_confirmation: bool = False
    confirmation_prompt: Optional[str] = None


class SecurityAuditLogger:
    """Structured security audit trail for all computer actions."""

    @staticmethod
    def _sanitize(data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact potential secrets, keys, or passwords from logs."""
        sensitive_keys = {"password", "secret", "token", "api_key", "authorization", "auth"}
        clean = {}
        for k, v in data.items():
            if any(s in k.lower() for s in sensitive_keys):
                clean[k] = "[REDACTED]"
            else:
                clean[k] = v
        return clean

    @classmethod
    def log_request(cls, tool_name: str, arguments: Dict[str, Any], risk: RiskLevel):
        safe_args = cls._sanitize(arguments)
        print(f"\033[90m[SECURITY] Tool request: {tool_name} (Risk: {risk.value})\033[0m", flush=True)
        logger.info(f"[SECURITY] Tool request: {tool_name} | Risk: {risk.value} | Args: {safe_args}")

    @classmethod
    def log_validation(cls, tool_name: str, passed: bool, reason: str = ""):
        status = "PASSED" if passed else "BLOCKED"
        color = "\033[92m" if passed else "\033[91m"
        print(f"{color}[SECURITY] Validation {status}: {tool_name} {('- ' + reason) if reason else ''}\033[0m", flush=True)
        if passed:
            logger.info(f"[SECURITY] Validation PASSED: {tool_name}")
        else:
            logger.warning(f"[SECURITY] Validation BLOCKED: {tool_name} | Reason: {reason}")

    @classmethod
    def log_execution(cls, tool_name: str, success: bool, details: str = ""):
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"[SECURITY] Execution {status}: {tool_name} | Details: {details}")


class ToolSecurityValidator:
    """
    Central security validation layer enforcing:
    1. Deny-by-default execution policy.
    2. Path sandboxing within JARVIS_WORKSPACE and allowed directories.
    3. Strict application and executable allowlisting.
    4. Safe URL validation (http/https only).
    5. High-risk confirmation checks.
    6. Emergency stop handling.
    """

    def __init__(self):
        self._state = ControlState.ENABLED
        self._workspace_root = Path(settings.jarvis_workspace).resolve()

    @property
    def control_state(self) -> ControlState:
        return self._state

    def is_enabled(self) -> bool:
        return self._state == ControlState.ENABLED and settings.computer_control_enabled

    def pause_control(self, reason: str = "Emergency stop"):
        """Emergency stop: Pause all computer control operations."""
        self._state = ControlState.PAUSED
        SecurityAuditLogger.log_validation("EMERGENCY_STOP", False, f"Computer control paused: {reason}")

    def resume_control(self):
        """Resume computer control operations."""
        self._state = ControlState.ENABLED
        logger.info("[SECURITY] Computer control resumed.")

    def get_allowed_roots(self) -> List[Path]:
        """Get all authorized directory roots."""
        roots = [self._workspace_root]
        for path_str in settings.allowed_directories:
            p = Path(path_str).resolve()
            if p.exists() and p not in roots:
                roots.append(p)
        return roots

    def validate_path(self, path_str: str) -> Path:
        """
        Validate and sandbox filesystem paths.
        - Resolves relative paths relative to JARVIS_WORKSPACE.
        - Resolves absolute paths and verifies they reside strictly inside an allowed root.
        - Rejects path traversal (e.g. `../../Windows`).
        """
        if not path_str or not str(path_str).strip():
            raise SecurityViolation("Path parameter cannot be empty.")

        # Block explicit suspicious substrings
        normalized_str = os.path.normpath(str(path_str).strip())
        if ".." in Path(normalized_str).parts:
            raise SecurityViolation(f"Path traversal detected in '{path_str}'. Access denied.")

        # Determine target path
        target_path = Path(normalized_str)
        if not target_path.is_absolute():
            target_path = (self._workspace_root / target_path).resolve()
        else:
            target_path = target_path.resolve()

        # Check containment in allowed roots
        allowed_roots = self.get_allowed_roots()
        is_inside_allowed = False
        for root in allowed_roots:
            try:
                # relative_to will succeed if target_path is inside root
                target_path.relative_to(root)
                is_inside_allowed = True
                break
            except ValueError:
                continue

        if not is_inside_allowed:
            raise SecurityViolation(
                f"Path '{target_path}' is outside the authorized workspace '{self._workspace_root}'. Access denied."
            )

        # Check system directory protections
        system_roots = [
            Path(os.environ.get("SystemRoot", "C:\\Windows")).resolve(),
            Path(os.environ.get("ProgramFiles", "C:\\Program Files")).resolve(),
            Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")).resolve(),
        ]
        for sys_root in system_roots:
            try:
                target_path.relative_to(sys_root)
                raise SecurityViolation(f"Access to protected system path '{target_path}' is strictly prohibited.")
            except ValueError:
                pass

        return target_path

    def validate_url(self, url_str: str) -> str:
        """
        Validate URLs for web browsing.
        Enforces http/https scheme and rejects local file://, javascript:, data:, shell: schemes.
        """
        if not url_str or not url_str.strip():
            raise SecurityViolation("URL cannot be empty.")

        url_clean = url_str.strip()
        parsed = urllib.parse.urlparse(url_clean)

        # If scheme is present and not http/https, reject immediately
        if parsed.scheme:
            if parsed.scheme.lower() not in ("http", "https"):
                raise SecurityViolation(
                    f"Unsupported URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted."
                )
        else:
            # If no scheme, check if user wrote something with colon like 'shell:System' or 'javascript:alert(1)'
            if ":" in url_clean:
                scheme_candidate = url_clean.split(":", 1)[0].lower()
                if scheme_candidate not in ("http", "https"):
                    raise SecurityViolation(
                        f"Unsupported URL scheme '{scheme_candidate}'. Only HTTP and HTTPS are permitted."
                    )
            # Otherwise it's a domain, prepend https://
            url_clean = f"https://{url_clean}"
            parsed = urllib.parse.urlparse(url_clean)

        if not parsed.netloc:
            raise SecurityViolation(f"Invalid URL target: '{url_str}'")

        return url_clean

    def is_confirmation_positive(self, user_response: str) -> bool:
        """
        Strictly verify user confirmation for high-risk actions.
        Rejects ambiguous or non-explicit responses.
        """
        if not user_response:
            return False

        cleaned = re.sub(r"[^\w\s]", "", user_response.strip().lower())
        words = cleaned.split()

        # Check if entire cleaned string matches a positive token
        if cleaned in POSITIVE_CONFIRMATION_TOKENS:
            return True

        # Check if first word is a clear positive token and no rejected words present
        if words and words[0] in {"yes", "confirm", "proceed", "affirmative", "authorized"}:
            if not any(rej in words for rej in REJECTED_CONFIRMATION_TOKENS):
                return True

        return False

    def validate_mouse_coordinates(
        self, x: Any, y: Any, screen_width: int, screen_height: int
    ) -> Tuple[int, int]:
        """
        Validate mouse screen coordinates against actual display bounds.
        Rejects non-numeric, NaN, Infinity, negative, or out-of-bounds coordinates.
        """
        import math

        if isinstance(x, bool) or isinstance(y, bool):
            raise SecurityViolation("Coordinates cannot be boolean values.")

        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise SecurityViolation(
                f"Coordinates must be numeric, got x={type(x).__name__}, y={type(y).__name__}"
            )

        if math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y):
            raise SecurityViolation("Coordinates cannot be NaN or Infinity.")

        target_x = int(round(x))
        target_y = int(round(y))

        if target_x < 0 or target_x >= screen_width or target_y < 0 or target_y >= screen_height:
            raise SecurityViolation(
                f"Coordinates ({target_x}, {target_y}) are out of screen bounds ({screen_width}x{screen_height})."
            )

        return target_x, target_y

    def validate_key(self, key_str: str) -> str:
        """
        Validate a single keyboard key against the safe allowlist.
        """
        if not key_str or not str(key_str).strip():
            raise SecurityViolation("Key parameter cannot be empty.")

        clean_key = str(key_str).strip().lower()
        if clean_key not in ALLOWED_KEYS:
            raise SecurityViolation(f"Key '{key_str}' is not in the authorized keyboard allowlist.")

        return clean_key

    def validate_hotkey(self, hotkey_str: str) -> Tuple[List[str], RiskLevel]:
        """
        Validate a keyboard shortcut combination against safe and dangerous allowlists.
        Returns the parsed key components and the assigned RiskLevel (LOW or HIGH).
        """
        if not hotkey_str or not str(hotkey_str).strip():
            raise SecurityViolation("Hotkey parameter cannot be empty.")

        raw_parts = [p.strip().lower() for p in re.split(r"[+\-\s]+", str(hotkey_str).strip()) if p.strip()]
        if not raw_parts:
            raise SecurityViolation("Invalid hotkey format.")

        # Validate each key component
        validated_keys = [self.validate_key(p) for p in raw_parts]
        combo_str = "+".join(validated_keys)

        # Check dangerous hotkeys (HIGH risk - require explicit user confirmation)
        if combo_str in DANGEROUS_HOTKEYS:
            return validated_keys, RiskLevel.HIGH

        # Check safe hotkeys
        if combo_str in SAFE_HOTKEYS:
            return validated_keys, RiskLevel.LOW

        # Allow basic modifier combinations (e.g. ctrl+a..z, alt+a..z, win+a..z)
        if len(validated_keys) == 2 and validated_keys[0] in {"ctrl", "control", "alt", "win", "shift"} and len(validated_keys[1]) == 1 and validated_keys[1].isalnum():
            return validated_keys, RiskLevel.LOW

        raise SecurityViolation(
            f"Hotkey '{hotkey_str}' is not in the authorized hotkey allowlist."
        )

    def is_protected_ui_element(self, label: str) -> bool:
        """
        Check if a UI element label indicates a sensitive or destructive action.
        """
        if not label:
            return False
        clean = label.strip().lower()
        return any(term in clean for term in PROTECTED_UI_ELEMENTS)


# Global security validator instance
security_validator = ToolSecurityValidator()

