import os
from pathlib import Path

try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False


def check_environment():
    results = []

    if not TORCH_INSTALLED:
        results.append({
            "status": "warn",
            "message": "PyTorch not found — skipping GPU checks",
            "fix": "Install PyTorch: https://pytorch.org/get-started/locally/"
        })
        return results

    # Check CUDA availability
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        results.append({
            "status": "pass",
            "message": f"CUDA available — {device_count} GPU(s) detected: {', '.join(device_names)}"
        })
    else:
        results.append({
            "status": "fail",
            "message": "CUDA not available — PyTorch cannot see any GPUs",
            "fix": "Check your CUDA installation and PyTorch version match"
        })

    # Check Accelerate config vs GPU count
    accelerate_results = check_accelerate()
    results.extend(accelerate_results)

    return results


def check_accelerate(config_path=None):
    results = []

    if not TORCH_INSTALLED:
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

        visible_gpus = torch.cuda.device_count()

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