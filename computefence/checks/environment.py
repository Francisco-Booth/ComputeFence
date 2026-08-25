import os
import platform
import sys
from pathlib import Path


def describe_interpreter():
    """Report the interpreter that is executing computefence right now.

    Every package check in this module imports in-process, so this path is
    exactly the environment those checks see. Surfacing it makes an
    install-location mismatch obvious instead of silent.
    """
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
    }


def in_active_virtualenv(interpreter):
    """True when VIRTUAL_ENV is set but computefence is running outside it.

    This is the failure that looks like a broken PyTorch check: the shell has a
    venv activated (with torch in it) while `computefence` resolves to a script
    installed against a different interpreter, which cannot see that torch.
    """
    virtual_env = interpreter["virtual_env"]
    if not virtual_env:
        return None
    try:
        return Path(interpreter["prefix"]).resolve() == Path(virtual_env).resolve()
    except OSError:
        return interpreter["prefix"] == virtual_env


def detect_torch():
    """Detect PyTorch with a direct, in-process import.

    The import runs inside the interpreter executing computefence, so it sees
    precisely what the active environment provides. No subprocess, no
    os.system, no hardcoded python path.
    """
    result = {
        "available": False,
        "version": None,
        "location": None,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "device_names": [],
        "error": None,
    }

    try:
        import torch
    except ImportError:
        return result
    except Exception as exc:
        # torch is installed but unusable — a CUDA/driver mismatch or a broken
        # build raises OSError or RuntimeError here, not ImportError. Reporting
        # it beats crashing the whole doctor run with a traceback.
        result["error"] = exc
        return result

    result["available"] = True
    result["version"] = getattr(torch, "__version__", "unknown")
    result["location"] = getattr(torch, "__file__", None)

    try:
        result["cuda_available"] = torch.cuda.is_available()
    except Exception as exc:
        result["error"] = exc
        return result

    if result["cuda_available"]:
        result["device_count"] = torch.cuda.device_count()
        result["device_names"] = [
            torch.cuda.get_device_name(i) for i in range(result["device_count"])
        ]
        if result["device_names"]:
            result["device_name"] = result["device_names"][0]

    return result


def check_environment():
    results = []

    interpreter = describe_interpreter()
    torch_info = detect_torch()

    results.append({
        "status": "pass",
        "message": f"Python {interpreter['version']} ({interpreter['executable']})"
    })

    # Warn when the shell has a venv active that computefence is not running in.
    if in_active_virtualenv(interpreter) is False:
        results.append({
            "status": "warn",
            "message": (
                f"computefence is not running inside the active virtual environment "
                f"({interpreter['virtual_env']}) — package checks below reflect "
                f"{interpreter['prefix']}, not your venv"
            ),
            "fix": "Install computefence into the active venv: pip install computefence"
        })

    if not torch_info["available"]:
        if torch_info["error"] is not None:
            results.append({
                "status": "fail",
                "message": f"PyTorch is installed but failed to import: {torch_info['error']}",
                "fix": "Check your PyTorch build matches the installed CUDA driver"
            })
        else:
            results.append({
                "status": "warn",
                "message": f"PyTorch not found in {interpreter['executable']} — skipping GPU checks",
                "fix": "Install PyTorch into this environment: https://pytorch.org/get-started/locally/"
            })
        return results

    if torch_info["cuda_available"]:
        results.append({
            "status": "pass",
            "message": (
                f"PyTorch {torch_info['version']} detected "
                f"(CUDA available — {', '.join(torch_info['device_names'])})"
            )
        })
    else:
        results.append({
            "status": "pass",
            "message": f"PyTorch {torch_info['version']} detected"
        })

    if torch_info["error"] is not None:
        results.append({
            "status": "fail",
            "message": f"Could not query CUDA: {torch_info['error']}",
            "fix": "Check your CUDA installation and PyTorch version match"
        })
        return results

    if not torch_info["cuda_available"]:
        results.append({
            "status": "fail",
            "message": "CUDA not available — PyTorch cannot see any GPUs",
            "fix": "Check your CUDA installation and PyTorch version match"
        })

    results.extend(check_accelerate(torch_info=torch_info))

    return results


def check_accelerate(config_path=None, torch_info=None):
    results = []

    if torch_info is None:
        torch_info = detect_torch()

    if not torch_info["available"]:
        return results

    # Find the Accelerate config
    if config_path:
        config_file = Path(config_path)
    else:
        # Try common default locations
        candidates = [
            Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml",
            Path.home() / ".config" / "huggingface" / "accelerate" / "default_config.yaml",
        ]

        # Also check HF_HOME based path
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidates.insert(0, Path(hf_home) / "accelerate" / "default_config.yaml")

        config_file = None
        for candidate in candidates:
            if candidate.exists():
                config_file = candidate
                break

    if config_file is None or not Path(config_file).exists():
        results.append({
            "status": "warn",
            "message": "No Accelerate config found — skipping GPU count check",
            "fix": "Run 'accelerate config' to create a configuration file"
        })
        return results

    # Parse the config
    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)

        num_processes = config.get("num_processes", None)

        if num_processes is None:
            results.append({
                "status": "warn",
                "message": "Could not read num_processes from Accelerate config",
                "fix": "Check your Accelerate config file is valid"
            })
            return results

        visible_gpus = torch_info["device_count"]

        if num_processes > visible_gpus:
            results.append({
                "status": "fail",
                "message": f"Accelerate config requests {num_processes} processes but only {visible_gpus} GPU(s) are visible",
                "fix": f"Run 'accelerate config' and set num_processes to {visible_gpus} or fewer"
            })
        else:
            results.append({
                "status": "pass",
                "message": f"Accelerate config requests {num_processes} process(es) — {visible_gpus} GPU(s) available"
            })

    except ImportError:
        results.append({
            "status": "warn",
            "message": "PyYAML not installed — cannot parse Accelerate config",
            "fix": "pip install pyyaml"
        })
    except Exception as e:
        results.append({
            "status": "warn",
            "message": f"Could not parse Accelerate config: {e}",
            "fix": "Check your Accelerate config file is valid YAML"
        })

    return results
