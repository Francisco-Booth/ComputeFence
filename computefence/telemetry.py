import hashlib
import platform
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

    warning_count = sum(1 for r in results if r.get("status") == "warn")
    blocker_count = sum(1 for r in results if r.get("status") == "fail")
    pass_count = sum(1 for r in results if r.get("status") == "pass")

    try:
        import torch
        gpu_count = torch.cuda.device_count()
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
    }

    thread = threading.Thread(target=_send, args=(payload,), daemon=True)
    thread.start()
    thread.join(timeout=4)