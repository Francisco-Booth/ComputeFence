"""
Drift detection for ComputeFence v0.2.13.

--save-env  : capture a 4-vector environment fingerprint after a clean run
              and write it to .computefence-lock.json in the persistent volume.
--check-drift : compare the current environment against a saved fingerprint
                and report differences as warnings (never blockers).
"""

import json
import os
from pathlib import Path


LOCK_FILENAME = ".computefence-lock.json"

PERSISTENT_VOLUME_CANDIDATES = [
    "/workspace",
    "/mnt/storage",
    "/runpod-volume",
    "/vast",
]


def _find_persistent_volume(path_override=None):
    if path_override:
        p = Path(path_override)
        if p.exists():
            return p
        return None
    for candidate in PERSISTENT_VOLUME_CANDIDATES:
        p = Path(candidate)
        if p.exists() and os.access(candidate, os.W_OK):
            return p
    return None


def _get_lock_path(path_override=None):
    # Primary: git repo root (project-level persistence)
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            return Path(result.stdout.strip()) / "computefence.lock"
    except Exception:
        pass
    # Fallback: persistent volume (pod-level persistence)
    vol = _find_persistent_volume(path_override)
    if vol is None:
        return None
    return vol / LOCK_FILENAME


def _vector_torch():
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "none",
        }
    except ImportError:
        return {"torch_version": "not_installed", "cuda_version": "none"}


def _vector_storage(output_dir=None):
    result = {}
    hf_home = os.environ.get("HF_HOME", "")
    result["hf_home"] = hf_home if hf_home else "unset"
    result["hf_home_persistent"] = any(
        hf_home.startswith(c) for c in PERSISTENT_VOLUME_CANDIDATES
    ) if hf_home else False
    result["output_dir"] = output_dir or "not_specified"
    result["output_dir_persistent"] = any(
        (output_dir or "").startswith(c) for c in PERSISTENT_VOLUME_CANDIDATES
    )
    return result


def _vector_gpu():
    try:
        import torch
        count = torch.cuda.device_count()
        caps = []
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            caps.append(f"{props.major}.{props.minor}")
        return {"gpu_count": count, "compute_capabilities": caps}
    except Exception:
        return {"gpu_count": 0, "compute_capabilities": []}


def _vector_libraries():
    libs = {}
    for lib in ["flash_attn", "transformers", "accelerate", "bitsandbytes"]:
        try:
            mod = __import__(lib)
            libs[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            libs[lib] = "not_installed"
    return libs


def capture_fingerprint(output_dir=None):
    return {
        "schema_version": 1,
        "torch": _vector_torch(),
        "storage": _vector_storage(output_dir),
        "gpu": _vector_gpu(),
        "libraries": _vector_libraries(),
    }


def save_env(output_dir=None, path_override=None):
    lock_path = _get_lock_path(path_override)
    if lock_path is None:
        return False, (
            "No persistent volume found. Cannot save environment fingerprint. "
            "Use --path /your/persistent/dir to specify one."
        )
    fingerprint = capture_fingerprint(output_dir)
    try:
        with open(lock_path, "w") as f:
            json.dump(fingerprint, f, indent=2)
        return True, str(lock_path)
    except Exception as e:
        return False, f"Failed to write {lock_path}: {e}"


def _diff_vectors(saved, current):
    results = []

    s_torch = saved.get("torch", {})
    c_torch = current.get("torch", {})
    if s_torch.get("torch_version") != c_torch.get("torch_version"):
        results.append({
            "status": "warn",
            "message": f"[Drift] PyTorch version changed: {s_torch.get('torch_version')} → {c_torch.get('torch_version')}",
            "fix": "Pin torch version in your requirements or Dockerfile.",
        })
    if s_torch.get("cuda_version") != c_torch.get("cuda_version"):
        results.append({
            "status": "warn",
            "message": f"[Drift] CUDA version changed: {s_torch.get('cuda_version')} → {c_torch.get('cuda_version')}",
            "fix": "Ensure your pod template uses the same CUDA base image.",
        })

    s_stor = saved.get("storage", {})
    c_stor = current.get("storage", {})
    if s_stor.get("hf_home") != c_stor.get("hf_home"):
        results.append({
            "status": "warn",
            "message": f"[Drift] HF_HOME changed: {s_stor.get('hf_home')} → {c_stor.get('hf_home')}",
            "fix": "Set HF_HOME consistently in your startup script.",
        })
    if s_stor.get("output_dir_persistent") and not c_stor.get("output_dir_persistent"):
        results.append({
            "status": "warn",
            "message": "[Drift] Output directory was persistent before — now appears ephemeral.",
            "fix": "Pass --output-dir pointing to /workspace or your persistent mount.",
        })

    s_gpu = saved.get("gpu", {})
    c_gpu = current.get("gpu", {})
    if s_gpu.get("gpu_count") != c_gpu.get("gpu_count"):
        results.append({
            "status": "warn",
            "message": f"[Drift] GPU count changed: {s_gpu.get('gpu_count')} → {c_gpu.get('gpu_count')}",
            "fix": "Check your pod configuration matches the saved environment.",
        })
    if s_gpu.get("compute_capabilities") != c_gpu.get("compute_capabilities"):
        results.append({
            "status": "warn",
            "message": f"[Drift] GPU compute capability changed: {s_gpu.get('compute_capabilities')} → {c_gpu.get('compute_capabilities')}",
            "fix": "Different GPU hardware detected. Verify model compatibility.",
        })

    s_libs = saved.get("libraries", {})
    c_libs = current.get("libraries", {})
    for lib in set(list(s_libs.keys()) + list(c_libs.keys())):
        sv = s_libs.get(lib, "not_installed")
        cv = c_libs.get(lib, "not_installed")
        if sv != cv:
            results.append({
                "status": "warn",
                "message": f"[Drift] {lib} version changed: {sv} → {cv}",
                "fix": f"Pin {lib} version in your requirements.",
            })

    if not results:
        results.append({
            "status": "pass",
            "message": "Environment matches saved fingerprint — no drift detected.",
        })

    return results


def check_drift(output_dir=None, path_override=None):
    lock_path = _get_lock_path(path_override)
    if lock_path is None:
        return [{"status": "warn", "message": "No persistent volume found.", "fix": "Use --path to specify your persistent directory."}]

    if not lock_path.exists():
        return [{"status": "warn", "message": f"No saved fingerprint at {lock_path}.", "fix": "Run: computefence doctor --save-env"}]

    try:
        with open(lock_path) as f:
            saved = json.load(f)
    except Exception as e:
        return [{"status": "warn", "message": f"Could not read fingerprint: {e}", "fix": "Delete the file and re-run: computefence doctor --save-env"}]

    current = capture_fingerprint(output_dir)
    return _diff_vectors(saved, current)
