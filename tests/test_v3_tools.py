"""
Functional Test Suite for JARVIS V3 Computer Control Tools.
Tests applications, system metrics, filesystem sandbox, screenshot, Windows UI, and media.
"""

import pytest
import asyncio
from pathlib import Path

from agent.config import settings
from agent.tools.registry import tool_registry
from agent.tools.router import tool_router
from agent.tools.applications import normalize_app_name, is_application_running
from agent.tools.system_info import get_system_metrics, get_gpu_status, get_running_applications
from agent.tools.filesystem import (
    create_folder,
    create_text_file,
    read_text_file,
    rename_file,
    move_file,
    copy_file,
    delete_file,
)
from agent.tools.screenshot import take_screenshot


@pytest.fixture
def sandbox_dir():
    """Create a temporary sandbox directory inside the workspace for filesystem tests."""
    sandbox = Path(settings.jarvis_workspace) / "data" / "test_sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    yield sandbox
    # Cleanup
    import shutil
    if sandbox.exists():
        shutil.rmtree(str(sandbox), ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. APPLICATION & INTENT MATCHING TESTS
# ---------------------------------------------------------------------------

def test_normalize_app_names():
    assert normalize_app_name("word") == "word"
    assert normalize_app_name("Microsoft Word") == "word"
    assert normalize_app_name("chrome") == "chrome"
    assert normalize_app_name("Google Chrome") == "chrome"
    assert normalize_app_name("calc") == "calculator"
    assert normalize_app_name("vs code") == "vscode"
    assert normalize_app_name("notepad") == "notepad"
    assert normalize_app_name("unknown_malicious_app") is None


def test_deterministic_intent_matching():
    assert tool_router.match_deterministic_intent("open word") == ("open_application", {"application": "word"})
    assert tool_router.match_deterministic_intent("launch chrome") == ("open_application", {"application": "chrome"})
    assert tool_router.match_deterministic_intent("start calculator") == ("open_application", {"application": "calculator"})
    assert tool_router.match_deterministic_intent("check my GPU") == ("get_gpu_status", {})
    assert tool_router.match_deterministic_intent("check my RAM") == ("get_system_metrics", {})
    assert tool_router.match_deterministic_intent("take a screenshot") == ("take_screenshot", {})
    assert tool_router.match_deterministic_intent("open settings") == ("open_settings", {})
    assert tool_router.match_deterministic_intent("volume up") == ("volume_up", {"steps": 2})
    assert tool_router.match_deterministic_intent("mute") == ("volume_mute", {})
    # Non-tools should return None
    assert tool_router.match_deterministic_intent("What is the capital of India?") is None
    assert tool_router.match_deterministic_intent("How are you today?") is None


# ---------------------------------------------------------------------------
# 2. SYSTEM TELEMETRY TESTS
# ---------------------------------------------------------------------------

def test_get_system_metrics():
    res = get_system_metrics()
    assert "CPU utilization" in res
    assert "RAM usage" in res
    assert "Disk storage" in res


def test_get_gpu_status():
    res = get_gpu_status()
    assert isinstance(res, str) and len(res) > 0


def test_get_running_applications():
    res = get_running_applications()
    assert isinstance(res, str) and len(res) > 0


# ---------------------------------------------------------------------------
# 3. FILESYSTEM SANDBOX CRUD TESTS
# ---------------------------------------------------------------------------

def test_filesystem_crud_pipeline(sandbox_dir):
    # 1. Create Folder
    folder_res = create_folder("data/test_sandbox/subfolder")
    assert "Folder 'subfolder' has been created" in folder_res

    # 2. Create File
    file_rel = "data/test_sandbox/sample.txt"
    create_res = create_text_file(file_rel, "Hello JARVIS V3 World!")
    assert "saved successfully" in create_res

    # 3. Read File
    read_res = read_text_file(file_rel)
    assert read_res == "Hello JARVIS V3 World!"

    # 4. Rename File
    rename_res = rename_file(file_rel, "renamed_sample.txt")
    assert "Renamed 'sample.txt' to 'renamed_sample.txt'" in rename_res

    # 5. Move File
    move_res = move_file("data/test_sandbox/renamed_sample.txt", "data/test_sandbox/subfolder")
    assert "Moved 'renamed_sample.txt' into 'subfolder'" in move_res

    # 6. Copy File
    copy_res = copy_file("data/test_sandbox/subfolder/renamed_sample.txt", "data/test_sandbox")
    assert "Copied 'renamed_sample.txt' into 'test_sandbox'" in copy_res

    # 7. Delete File with Confirmation
    del_res = delete_file("data/test_sandbox/renamed_sample.txt", confirmed=True)
    assert "has been deleted" in del_res


# ---------------------------------------------------------------------------
# 4. SCREENSHOT TEST
# ---------------------------------------------------------------------------

def test_take_screenshot():
    res = take_screenshot("test_v3_run.png")
    assert "Screenshot captured" in res
    target = Path(settings.jarvis_workspace) / "data" / "screenshots" / "test_v3_run.png"
    assert target.exists()
    # Cleanup test screenshot
    target.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 5. SECURE REGISTRY DISPATCH TEST
# ---------------------------------------------------------------------------

def test_tool_registry_execute_secure():
    result = asyncio.run(tool_registry.execute_secure("get_system_metrics"))
    assert "CPU utilization" in result
    assert "RAM usage" in result
