import os
from pathlib import Path


def check_storage():
    results = []

    # Read HuggingFace cache environment variables
    hf_home = os.environ.get("HF_HOME")
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")

    # Detect mounted persistent volumes
    mounted_volumes = []
    for path in ["/workspace", "/runpod-volume", "/vast"]:
        if Path(path).exists():
            mounted_volumes.append(path)

    # Check HF_HOME
    if hf_home is None:
        if mounted_volumes:
            results.append({
                "status": "warn",
                "message": f"HF_HOME is not set. Mounted volume detected at {mounted_volumes[0]} but HuggingFace will cache to default local path.",
                "fix": f"export HF_HOME={mounted_volumes[0]}/.cache/huggingface"
            })
        else:
            results.append({
                "status": "warn",
                "message": "HF_HOME is not set. HuggingFace will use default local cache.",
                "fix": "Set HF_HOME to a persistent volume path before training"
            })
    elif hf_home.startswith("/root") or hf_home.startswith("/tmp"):
        results.append({
            "status": "warn",
            "message": f"HF_HOME appears to point to ephemeral storage: {hf_home}",
            "fix": f"export HF_HOME=/workspace/.cache/huggingface"
        })
    elif mounted_volumes and not any(hf_home.startswith(v) for v in mounted_volumes):
        results.append({
            "status": "warn",
            "message": f"HF_HOME ({hf_home}) does not appear to be on a mounted volume.",
            "fix": f"export HF_HOME={mounted_volumes[0]}/.cache/huggingface"
        })
    else:
        results.append({
            "status": "pass",
            "message": f"HF_HOME is set to {hf_home}"
        })

    # Report mounted volumes found
    if mounted_volumes:
        results.append({
            "status": "pass",
            "message": f"Persistent volume(s) detected: {', '.join(mounted_volumes)}"
        })
    else:
        results.append({
            "status": "warn",
            "message": "No mounted persistent volumes detected at /workspace, /runpod-volume, or /vast",
            "fix": "Attach a network volume before training to persist checkpoints and cache"
        })

    return results