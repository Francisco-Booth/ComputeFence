import hashlib
import os
import platform
import subprocess
import sys
import threading
import uuid
from pathlib import Path

SUPABASE_URL = "https://pidpadpudbcdldyvrlog.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_A2zwbfTFiOs63jz8313GXg_elw3F5dy"
TABLE = "computefence_runs"
OPT_OUT_FILE = Path.home() / ".computefence_no_telemetry"

def _is_opted_out():
    return OPT_OUT_FILE.exists()

def get_project_id() -> str:
    # Primary: persistent home directory token — not linked to any repo
    home_token_path = Path.home() / ".computefence" / "id"
    try:
        if home_token_path.exists():
            token = home_token_path.read_text().strip()
            if token:
                return token
        home_token_path.parent.mkdir(parents=True, exist_ok=True)
        new_token = str(uuid.uuid4())
        home_token_path.write_text(new_token)
        return new_token
    except Exception:
        pass

    # Fallback: persistent volume token at /workspace
    token_path = "/workspace/.computefence_id"
    try:
        if os.path.exists(token_path):
            with open(token_path) as f:
                token = f.read().strip()
                if token:
                    return token
        new_token = str(uuid.uuid4())
        with open(token_path, "w") as f:
            f.write(new_token)
        return new_token
    except Exception:
        pass

    # Last resort: ephemeral cwd hash
    return "ephemeral-" + hashlib.sha256(os.getcwd().encode()).hexdigest()[:8]

def get_execution_mode() -> str:
    if os.environ.get("GITHUB_ACTIONS"):
        return "github-action"
    if os.environ.get("CI"):
        return "ci"
    try:
        ppid = os.getppid()
        result = subprocess.run(
            ["ps", "-p", str(ppid), "-o", "comm="],
            capture_output=True, text=True, timeout=2
        )
        parent = result.stdout.strip().lower()
        if "bash" in parent or "sh" in parent:
            cmdline_path = f"/proc/{ppid}/cmdline"
            if os.path.exists(cmdline_path):
                with open(cmdline_path, "rb") as f:
                    cmdline = f.read().decode(errors="replace").replace("\x00", " ")
                if "on-start" in cmdline or "startup" in cmdline or "entrypoint" in cmdline:
                    return "on-start"
    except Exception:
        pass
    return "manual"

def _send(payload: dict):
    try:
        import requests
        requests.post(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=payload,
            timeout=3,
        )
    except Exception:
        pass

def record_run(results: list):
    if _is_opted_out():
        return
    if os.environ.get("COMPUTEFENCE_DEV"):
        return
    warning_count = sum(1 for r in results if r.get("status") == "warn")
    blocker_count = sum(1 for r in results if r.get("status") == "fail")
    pass_count = sum(1 for r in results if r.get("status") == "pass")
    try:
        import torch
        gpu_count = torch.cuda.device_count()
    except Exception:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            gpu_count = len(result.stdout.strip().splitlines()) if result.returncode == 0 else 0
        except Exception:
            gpu_count = 0
    from computefence import __version__
    payload = {
        "version": __version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.system(),
        "gpu_count": gpu_count,
        "warning_count": warning_count,
        "blocker_count": blocker_count,
        "pass_count": pass_count,
        "opted_in": True,
        "project_id": get_project_id(),
        "run_id": str(uuid.uuid4()),
        "execution_mode": get_execution_mode(),
    }
    thread = threading.Thread(target=_send, args=(payload,), daemon=True)
    thread.start()
    thread.join(timeout=4)
