"""
System telemetry and hardware monitoring tools for JARVIS V3.
Gathers CPU, RAM, GPU, Disk, Battery, and running application metrics.
"""

import logging
import subprocess
from typing import Dict, Any, List
import psutil

from agent.tools.security import SecurityAuditLogger

logger = logging.getLogger("jarvis.tools.system_info")


def get_system_metrics() -> str:
    """
    Retrieve current CPU, RAM, Disk, and Battery metrics.
    """
    try:
        # CPU
        cpu_pct = psutil.cpu_percent(interval=0.2)
        cpu_count = psutil.cpu_count(logical=True)

        # RAM
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024 ** 3)
        ram_total_gb = ram.total / (1024 ** 3)
        ram_pct = ram.percent

        # Disk (Primary workspace disk)
        disk = psutil.disk_usage("D:\\" if psutil.os.path.exists("D:\\") else "C:\\")
        disk_free_gb = disk.free / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)

        # Battery
        battery_str = ""
        battery = psutil.sensors_battery()
        if battery:
            status = "charging" if battery.power_plugged else "on battery"
            battery_str = f" Battery is at {battery.percent:.0f}% ({status})."

        summary = (
            f"CPU utilization is at {cpu_pct:.0f}% across {cpu_count} logical cores. "
            f"RAM usage is {ram_pct:.0f}% ({ram_used_gb:.1f} GB of {ram_total_gb:.1f} GB). "
            f"Disk storage has {disk_free_gb:.1f} GB free of {disk_total_gb:.1f} GB.{battery_str}"
        )

        SecurityAuditLogger.log_execution("get_system_metrics", True, summary)
        return summary
    except Exception as e:
        SecurityAuditLogger.log_execution("get_system_metrics", False, str(e))
        return f"Unable to retrieve system metrics: {e}"


def get_gpu_status() -> str:
    """
    Retrieve NVIDIA GPU utilization, VRAM usage, and temperature.
    Uses nvidia-smi with direct arguments (NO shell=True).
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            line = res.stdout.strip()
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                name, gpu_util, mem_used, mem_total, temp = parts[:5]
                used_gb = float(mem_used) / 1024
                total_gb = float(mem_total) / 1024
                summary = (
                    f"Your {name} is at {gpu_util}% utilization, with "
                    f"{used_gb:.2f} GB of {total_gb:.2f} GB VRAM in use, running at {temp}°C."
                )
                SecurityAuditLogger.log_execution("get_gpu_status", True, summary)
                return summary

        return "NVIDIA GPU is active, but detailed metrics could not be queried."
    except FileNotFoundError:
        return "nvidia-smi is not found in system PATH. GPU telemetry is currently unavailable."
    except Exception as e:
        SecurityAuditLogger.log_execution("get_gpu_status", False, str(e))
        return f"Unable to query GPU status: {e}"


def get_running_applications() -> str:
    """
    List user-facing applications currently running on the system.
    Filters out background system services.
    """
    try:
        user_apps = set()
        # Common GUI process names
        known_gui_apps = {
            "chrome.exe": "Google Chrome",
            "msedge.exe": "Microsoft Edge",
            "code.exe": "Visual Studio Code",
            "winword.exe": "Microsoft Word",
            "excel.exe": "Microsoft Excel",
            "powerpnt.exe": "Microsoft PowerPoint",
            "notepad.exe": "Notepad",
            "calc.exe": "Calculator",
            "spotify.exe": "Spotify",
            "discord.exe": "Discord",
            "slack.exe": "Slack",
            "teams.exe": "Microsoft Teams",
            "taskmgr.exe": "Task Manager",
            "explorer.exe": "File Explorer",
            "antigravity ide.exe": "Antigravity IDE",
        }

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = proc.info["name"]
                if name:
                    name_lower = name.lower()
                    if name_lower in known_gui_apps:
                        user_apps.add(known_gui_apps[name_lower])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if user_apps:
            apps_list = ", ".join(sorted(user_apps))
            summary = f"Currently active applications include: {apps_list}."
        else:
            summary = "No major user applications are currently active."

        SecurityAuditLogger.log_execution("get_running_applications", True, summary)
        return summary
    except Exception as e:
        SecurityAuditLogger.log_execution("get_running_applications", False, str(e))
        return f"Unable to retrieve running applications: {e}"
