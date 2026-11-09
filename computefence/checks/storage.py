import os
import shutil
from pathlib import Path

MOUNT_CANDIDATES = ["/workspace", "/runpod-volume", "/vast"]

GB = 1024 ** 3
BLOCKER_BELOW_GB = 5
WARNING_BELOW_GB = 20
DISK_FIX = "Free up disk space or move checkpoints to a larger volume: df -h to check usage"


def check_storage():
    results = []

    hf_home = os.environ.get("HF_HOME")

    mounted_volumes = []
    for path in MOUNT_CANDIDATES:
        if Path(path).exists():
            mounted_volumes.append(path)

    if hf_home is None:
        if mounted_volumes:
            results.append({
                "id": "hf_home",
                "status": "warn",
                "message": f"HF_HOME is not set. Mounted volume detected at {mounted_volumes[0]} but HuggingFace will cache to default local path.",
                "fix": f"export HF_HOME={mounted_volumes[0]}/.cache/huggingface"
            })
        else:
            results.append({
                "id": "hf_home",
                "status": "warn",
                "message": "HF_HOME is not set. HuggingFace will use default local cache.",
                "fix": "export HF_HOME=/workspace/.cache/huggingface"
            })
    elif hf_home.startswith("/root") or hf_home.startswith("/tmp"):
        results.append({
            "id": "hf_home",
            "status": "warn",
            "message": f"HF_HOME appears to point to ephemeral storage: {hf_home}",
            "fix": "export HF_HOME=/workspace/.cache/huggingface"
        })
    elif mounted_volumes and not any(hf_home.startswith(v) for v in mounted_volumes):
        results.append({
            "id": "hf_home",
            "status": "warn",
            "message": f"HF_HOME ({hf_home}) does not appear to be on a mounted volume.",
            "fix": f"export HF_HOME={mounted_volumes[0]}/.cache/huggingface"
        })
    else:
        results.append({
            "id": "hf_home",
            "status": "pass",
            "message": f"HF_HOME is set to {hf_home}"
        })

    if mounted_volumes:
        results.append({
            "id": "persistent_volume",
            "status": "pass",
            "message": f"Persistent volume(s) detected: {', '.join(mounted_volumes)}"
        })
    else:
        results.append({
            "id": "persistent_volume",
            "status": "warn",
            "message": "No mounted persistent volumes detected at /workspace, /runpod-volume, or /vast",
            "fix": "Mount a network volume at /workspace before launching your job"
        })

    return results


def check_disk_headroom():
    results = []

    targets = []
    seen_devices = set()
    for path in MOUNT_CANDIDATES + ["/"]:
        candidate = Path(path)
        if not candidate.exists():
            continue
        try:
            device = candidate.stat().st_dev
        except OSError:
            device = None
        if device is not None:
            if device in seen_devices:
                continue
            seen_devices.add(device)
        targets.append(candidate)

    for target in targets:
        is_root = str(target) == "/"
        label = "Root disk (/)" if is_root else f"Volume {target}"
        check_id = "disk_root" if is_root else f"disk_{str(target).strip('/').replace('/', '_')}"

        try:
            usage = shutil.disk_usage(target)
        except OSError as e:
            results.append({
                "id": check_id,
                "status": "warn",
                "message": f"{label} — could not read disk usage: {e}",
                "fix": DISK_FIX
            })
            continue

        free_gb = usage.free / GB
        total_gb = usage.total / GB
        summary = f"{label} — {free_gb:.1f} GB free of {total_gb:.1f} GB"

        if free_gb < BLOCKER_BELOW_GB:
            results.append({
                "id": check_id,
                "status": "fail",
                "message": f"{summary} (below {BLOCKER_BELOW_GB} GB)",
                "fix": DISK_FIX
            })
        elif free_gb < WARNING_BELOW_GB:
            results.append({
                "id": check_id,
                "status": "warn",
                "message": f"{summary} (below {WARNING_BELOW_GB} GB)",
                "fix": DISK_FIX
            })
        else:
            results.append({
                "id": check_id,
                "status": "pass",
                "message": summary
            })

    return results


EPHEMERAL_PREFIXES = ["/root", "/tmp"]
PERSISTENT_PREFIXES = ["/workspace", "/runpod-volume", "/vast", "/home"]
OUTPUT_DIR_FIX = "Move your output_dir to /workspace/checkpoints or your mounted volume"


def _is_under(path, prefix):
    path_parts = Path(path).parts
    prefix_parts = Path(prefix).parts
    return path_parts[: len(prefix_parts)] == prefix_parts


def check_output_dir(output_dir=None):
    if output_dir is None:
        return []

    path = Path(os.path.abspath(os.path.expanduser(str(output_dir))))

    is_ephemeral = any(_is_under(path, prefix) for prefix in EPHEMERAL_PREFIXES) or not any(
        _is_under(path, prefix) for prefix in PERSISTENT_PREFIXES
    )

    if is_ephemeral:
        return [{
            "id": "output_dir",
            "status": "fail",
            "message": f"Output directory {path} is on ephemeral storage and will be lost when the instance stops",
            "fix": OUTPUT_DIR_FIX
        }]

    return [{
        "id": "output_dir",
        "status": "pass",
        "message": f"Output directory {path} is on persistent storage"
    }]