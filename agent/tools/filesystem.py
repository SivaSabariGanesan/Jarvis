"""
Sandboxed filesystem tools for JARVIS V3.
Enforces strict workspace directory isolation and explicit confirmation for deletions.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional

from agent.tools.security import (
    security_validator,
    RiskLevel,
    SecurityViolation,
    ConfirmationRequired,
    SecurityAuditLogger,
)

logger = logging.getLogger("jarvis.tools.filesystem")


def create_folder(folder_path: str) -> str:
    """
    Create a new directory within the authorized workspace.
    """
    target = security_validator.validate_path(folder_path)
    try:
        target.mkdir(parents=True, exist_ok=True)
        SecurityAuditLogger.log_execution("create_folder", True, f"Created directory {target}")
        return f"Folder '{target.name}' has been created in your workspace, sir."
    except Exception as e:
        SecurityAuditLogger.log_execution("create_folder", False, str(e))
        return f"Unable to create folder: {e}"


def create_text_file(file_path: str, content: str) -> str:
    """
    Create or write a text file within the authorized workspace.
    """
    target = security_validator.validate_path(file_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        SecurityAuditLogger.log_execution("create_text_file", True, f"Wrote file {target} ({len(content)} chars)")
        return f"File '{target.name}' has been saved successfully in your workspace, sir."
    except Exception as e:
        SecurityAuditLogger.log_execution("create_text_file", False, str(e))
        return f"Unable to write file: {e}"


def read_text_file(file_path: str, max_lines: int = 100) -> str:
    """
    Read the contents of a text file from the authorized workspace.
    """
    target = security_validator.validate_path(file_path)
    if not target.exists():
        raise FileNotFoundError(f"File '{target.name}' does not exist.")
    if not target.is_file():
        raise ValueError(f"'{target.name}' is a directory, not a file.")

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(max_lines)]
            content = "".join(lines)
            if f.readline():
                content += f"\n... [Truncated after {max_lines} lines]"

        SecurityAuditLogger.log_execution("read_text_file", True, f"Read {target}")
        return content if content else "[File is empty]"
    except Exception as e:
        SecurityAuditLogger.log_execution("read_text_file", False, str(e))
        return f"Unable to read file: {e}"


def rename_file(source_path: str, target_name: str) -> str:
    """
    Rename a file or folder inside the authorized workspace.
    target_name must be a simple filename without path separators.
    """
    source = security_validator.validate_path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source '{source.name}' does not exist.")

    # Target name must be a clean name
    clean_target_name = os.path.basename(target_name.strip())
    if not clean_target_name or "/" in target_name or "\\" in target_name:
        raise SecurityViolation("Target name must be a single file/folder name, not a path.")

    dest = source.parent / clean_target_name
    dest = security_validator.validate_path(str(dest))

    try:
        source.rename(dest)
        SecurityAuditLogger.log_execution("rename_file", True, f"Renamed {source} -> {dest}")
        return f"Renamed '{source.name}' to '{dest.name}', sir."
    except Exception as e:
        SecurityAuditLogger.log_execution("rename_file", False, str(e))
        return f"Unable to rename file: {e}"


def move_file(source_path: str, target_folder: str) -> str:
    """
    Move a file to another folder within the authorized workspace.
    """
    source = security_validator.validate_path(source_path)
    folder = security_validator.validate_path(target_folder)

    if not source.exists():
        raise FileNotFoundError(f"Source file '{source.name}' does not exist.")
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Destination folder '{folder.name}' does not exist.")

    dest = folder / source.name
    dest = security_validator.validate_path(str(dest))

    try:
        shutil.move(str(source), str(dest))
        SecurityAuditLogger.log_execution("move_file", True, f"Moved {source} -> {dest}")
        return f"Moved '{source.name}' into '{folder.name}', sir."
    except Exception as e:
        SecurityAuditLogger.log_execution("move_file", False, str(e))
        return f"Unable to move file: {e}"


def copy_file(source_path: str, target_folder: str) -> str:
    """
    Copy a file to another folder within the authorized workspace.
    """
    source = security_validator.validate_path(source_path)
    folder = security_validator.validate_path(target_folder)

    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Source file '{source.name}' does not exist.")
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Destination folder '{folder.name}' does not exist.")

    dest = folder / source.name
    dest = security_validator.validate_path(str(dest))

    try:
        shutil.copy2(str(source), str(dest))
        SecurityAuditLogger.log_execution("copy_file", True, f"Copied {source} -> {dest}")
        return f"Copied '{source.name}' into '{folder.name}', sir."
    except Exception as e:
        SecurityAuditLogger.log_execution("copy_file", False, str(e))
        return f"Unable to copy file: {e}"


def delete_file(file_path: str, confirmed: bool = False) -> str:
    """
    Delete a file within the authorized workspace.
    HIGH_RISK: Requires explicit confirmation before deletion.
    """
    target = security_validator.validate_path(file_path)
    if not target.exists():
        raise FileNotFoundError(f"File '{target.name}' does not exist.")

    if not confirmed:
        raise ConfirmationRequired(
            f"Deleting '{target.name}' will permanently remove this file from your workspace. Do you want me to proceed, sir?",
            tool_name="delete_file",
            arguments={"file_path": file_path, "confirmed": True},
        )

    try:
        if target.is_dir():
            shutil.rmtree(str(target))
        else:
            target.unlink()
        SecurityAuditLogger.log_execution("delete_file", True, f"Deleted {target}")
        return f"'{target.name}' has been deleted from your workspace, sir."
    except Exception as e:
        SecurityAuditLogger.log_execution("delete_file", False, str(e))
        return f"Unable to delete '{target.name}': {e}"
