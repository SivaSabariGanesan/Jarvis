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


# Global security validator instance
security_validator = ToolSecurityValidator()
