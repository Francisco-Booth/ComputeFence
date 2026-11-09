import os
import platform
import sys
from pathlib import Path


def describe_interpreter():
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
    }


def in_active_virtualenv(interpreter):
    virtual_env = interpreter["virtual_env"]
    if not virtual_env:
        return None
    try:
        return Path(interpreter["prefix"]).resolve() == Path(virtual_env).resolve()
    except OSError:
        return interpreter["prefix"] == virtual_env


def detect_torch():
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


def check_gpu_arch_compat(torch_info):
    """Check that each visible GPU's compute capability is supported by this PyTorch build."""
    results = []
    try:
        import torch
        if not torch_info["cuda_available"]:
            return results

        try:
            supported = torch.cuda.get_arch_list()
        except Exception:
            return results

        # Parse supported list: ["sm_50", "sm_60", ...] → {(5,0), (6,0), ...}
        supported_caps = set()
        for arch in supported:
            if arch.startswith("sm_") and len(arch) > 3:
                digits = arch[3:]
                if len(digits) >= 2:
                    try:
                        major = int(digits[:-1])
                        minor = int(digits[-1])
                        supported_caps.add((major, minor))
                    except ValueError:
                        pass

        if not supported_caps:
            return results

        max_supported = max(supported_caps)

        for i in range(torch_info["device_count"]):
            name = torch_info["device_names"][i] if i < len(torch_info["device_names"]) else f"GPU {i}"
            try:
                cap = torch.cuda.get_device_capability(i)
            except Exception:
                continue

            # GPU is compatible if its exact capability is listed, or if it's
            # within the same major version and not newer than max supported
            compatible = (cap in supported_caps) or (
                cap[0] == max_supported[0] and cap <= max_supported
            )

            if not compatible:
                supported_str = ", ".join(
                    f"sm_{m}{n}" for m, n in sorted(supported_caps)
                )
                results.append({
                    "id": f"gpu_arch_compat_{i}",
                    "status": "fail",
                    "message": (
                        f"{name} has compute capability sm_{cap[0]}{cap[1]}, "
                        f"which is not supported by this PyTorch build "
                        f"(supports up to sm_{max_supported[0]}{max_supported[1]})"
                    ),
                    "fix": (
                        f"Install a PyTorch build that supports sm_{cap[0]}{cap[1]}: "
                        f"see https://pytorch.org/get-started/locally/"
                    ),
                })
            else:
                results.append({
                    "id": f"gpu_arch_compat_{i}",
                    "status": "pass",
                    "message": (
                        f"{name} compute capability sm_{cap[0]}{cap[1]} "
                        f"is supported by this PyTorch build"
                    ),
                })

    except Exception:
        pass

    return results


def check_environment():
    results = []

    interpreter = describe_interpreter()
    torch_info = detect_torch()

    results.append({
        "id": "python_version",
        "status": "pass",
        "message": f"Python {interpreter['version']} ({interpreter['executable']})"
    })

    if in_active_virtualenv(interpreter) is False:
        results.append({
            "id": "virtualenv_mismatch",
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
                "id": "torch_import",
                "status": "fail",
                "message": f"PyTorch is installed but failed to import: {torch_info['error']}",
                "fix": "Check your PyTorch build matches the installed CUDA driver"
            })
        else:
            results.append({
                "id": "torch_missing",
                "status": "warn",
                "message": f"PyTorch not found in {interpreter['executable']} — skipping GPU checks",
                "fix": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            })
        return results

    if torch_info["cuda_available"]:
        results.append({
            "id": "torch_cuda",
            "status": "pass",
            "message": (
                f"PyTorch {torch_info['version']} detected "
                f"(CUDA available — {', '.join(torch_info['device_names'])})"
            )
        })
    else:
        results.append({
            "id": "torch_cuda",
            "status": "pass",
            "message": f"PyTorch {torch_info['version']} detected"
        })

    if torch_info["error"] is not None:
        results.append({
            "id": "cuda_query",
            "status": "fail",
            "message": f"Could not query CUDA: {torch_info['error']}",
            "fix": "Check your CUDA installation and PyTorch version match"
        })
        return results

    if not torch_info["cuda_available"]:
        results.append({
            "id": "cuda_visibility",
            "status": "fail",
            "message": "CUDA not available — PyTorch cannot see any GPUs",
            "fix": "Check your CUDA installation and PyTorch version match"
        })
    else:
        results.extend(check_gpu_arch_compat(torch_info))

    results.extend(check_accelerate(torch_info=torch_info))

    return results


def check_accelerate(config_path=None, torch_info=None):
    results = []

    if torch_info is None:
        torch_info = detect_torch()

    if not torch_info["available"]:
        return results

    if config_path:
        config_file = Path(config_path)
    else:
        candidates = [
            Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml",
            Path.home() / ".config" / "huggingface" / "accelerate" / "default_config.yaml",
        ]
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
            "id": "accelerate_config",
            "status": "warn",
            "message": "No Accelerate config found — skipping GPU count check",
            "fix": "Run 'accelerate config' to create a configuration file"
        })
        return results

    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)

        num_processes = config.get("num_processes", None)

        if num_processes is None:
            results.append({
                "id": "accelerate_num_processes",
                "status": "warn",
                "message": "Could not read num_processes from Accelerate config",
                "fix": "Check your Accelerate config file is valid"
            })
            return results

        visible_gpus = torch_info["device_count"]

        if num_processes > visible_gpus:
            results.append({
                "id": "accelerate_gpu_count",
                "status": "fail",
                "message": f"Accelerate config requests {num_processes} processes but only {visible_gpus} GPU(s) are visible",
                "fix": "Run accelerate config to update your GPU count then re-run doctor"
            })
        else:
            results.append({
                "id": "accelerate_gpu_count",
                "status": "pass",
                "message": f"Accelerate config requests {num_processes} process(es) — {visible_gpus} GPU(s) available"
            })

    except ImportError:
        results.append({
            "id": "accelerate_pyyaml",
            "status": "warn",
            "message": "PyYAML not installed — cannot parse Accelerate config",
            "fix": "pip install pyyaml"
        })
    except Exception as e:
        results.append({
            "id": "accelerate_parse",
            "status": "warn",
            "message": f"Could not parse Accelerate config: {e}",
            "fix": "Check your Accelerate config file is valid YAML"
        })

    return results
