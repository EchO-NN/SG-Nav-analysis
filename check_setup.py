#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def status(label, ok, detail=""):
    mark = "OK" if ok else "WARN"
    suffix = f" - {detail}" if detail else ""
    print(f"[{mark}] {label}{suffix}")


def required_path(rel_path, description):
    path = ROOT / rel_path
    ok = path.exists()
    status(description, ok, rel_path if ok else f"missing: {rel_path}")
    return ok


def run(cmd, timeout=20, env_lib=None):
    env = os.environ.copy()
    env["NLTK_DATA"] = str(ROOT / ".cache/nltk_data")
    env["HF_HOME"] = str(ROOT / ".cache/huggingface")
    env["PYTHONPATH"] = (
        f"{ROOT / 'habitat-lab'}:{ROOT / 'GroundingDINO'}:"
        f"{ROOT / 'GLIP'}:{ROOT}:{env.get('PYTHONPATH', '')}"
    ).rstrip(":")
    lib_paths = [
        ROOT / ".mamba/envs/sg-nav/lib/python3.9/site-packages/torch/lib",
        ROOT / ".mamba/envs/sg-nav/lib",
    ]
    cuda12_runtime = Path("/usr/local/lib/ollama/cuda_v12")
    portable_cuda12_runtime = ROOT / ".cuda_v12"
    if portable_cuda12_runtime.exists():
        lib_paths.insert(0, portable_cuda12_runtime)
    if cuda12_runtime.exists():
        lib_paths.insert(0, cuda12_runtime)
    if env_lib:
        lib_paths.insert(0, ROOT / env_lib)
    old_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in lib_paths)
    if old_ld_path:
        env["LD_LIBRARY_PATH"] = f"{env['LD_LIBRARY_PATH']}:{old_ld_path}"
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)
    return completed.returncode == 0, completed.stdout.strip()


def compact_output(output, prefixes):
    lines = []
    for line in output.splitlines():
        line = line.strip()
        if any(line.startswith(prefix) for prefix in prefixes):
            lines.append(line)
    if lines:
        return "; ".join(lines)
    return output.replace("\n", "; ") if output else ""


def check_main_env():
    python = ROOT / ".mamba/envs/sg-nav/bin/python"
    if not required_path(".mamba/envs/sg-nav/bin/python", "SG-Nav mamba env"):
        return False
    ok, output = run([
        str(python),
        "-c",
        (
            "import torch, habitat, requests; "
            "from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo; "
            "import maskrcnn_benchmark._C; "
            "import maskrcnn_benchmark; "
            "print('python ok'); "
            "print('torch=' + torch.__version__); "
            "print('cuda=' + str(torch.cuda.is_available())); "
            "print('glip_path=' + maskrcnn_benchmark.__file__); "
            "print('glip_ext=ok'); "
            "print('glip_predictor=ok')"
        ),
    ], env_lib=".mamba/envs/sg-nav/lib")
    status("SG-Nav imports", ok, compact_output(output, ("python ok", "torch=", "cuda=", "glip_path=", "glip_ext=", "glip_predictor=")))
    return ok


def check_vllm_env():
    python = ROOT / ".mamba/envs/sg-nav-vllm/bin/python"
    vllm = ROOT / ".mamba/envs/sg-nav-vllm/bin/vllm"
    env_ok = required_path(".mamba/envs/sg-nav-vllm/bin/python", "vLLM mamba env")
    cli_ok = required_path(".mamba/envs/sg-nav-vllm/bin/vllm", "vLLM CLI")
    if not env_ok:
        return False
    ok, output = run([
        str(python),
        "-c",
        (
            "import importlib.util, vllm, torch; "
            "print('vllm=' + vllm.__version__); "
            "print('torch=' + torch.__version__); "
            "print('cuda=' + str(torch.cuda.is_available())); "
            "print('qwen3_vl=' + str(bool(importlib.util.find_spec('vllm.model_executor.models.qwen3_vl'))))"
        ),
    ], env_lib=".mamba/envs/sg-nav-vllm/lib")
    status("vLLM imports", ok, compact_output(output, ("vllm=", "torch=", "cuda=", "qwen3_vl=")))
    return ok and cli_ok


def check_assets():
    checks = [
        ("data/models/sam_vit_h_4b8939.pth", "SAM checkpoint"),
        ("data/models/groundingdino_swint_ogc.pth", "GroundingDINO checkpoint"),
        ("GLIP/MODEL/glip_large_model.pth", "GLIP checkpoint"),
        ("data/MatterPort3D/objectnav/mp3d/v1/val/val.json.gz", "ObjectNav val episodes"),
    ]
    ok = all(required_path(path, label) for path, label in checks)

    scene_files = sorted((ROOT / "data/MatterPort3D/mp3d").glob("*/*.glb"))
    content_files = sorted((ROOT / "data/MatterPort3D/objectnav/mp3d/v1/val/content").glob("*.json.gz"))
    status("MatterPort3D scene files", bool(scene_files), f"{len(scene_files)} .glb files")
    status("ObjectNav val content files", bool(content_files), f"{len(content_files)} scene episode files")
    return ok and bool(scene_files) and bool(content_files)


def check_gpu():
    ok, output = run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ], timeout=10)
    status("nvidia-smi", ok, output.replace("\n", "; ") if output else "not visible")
    return ok


def check_vllm_server():
    base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    models_url = f"{base_url}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as response:
            payload = response.read().decode("utf-8", errors="replace")
        model_ids = []
        try:
            data = json.loads(payload)
            model_ids = [item.get("id", "") for item in data.get("data", []) if item.get("id")]
        except json.JSONDecodeError:
            pass
        detail = ", ".join(model_ids) if model_ids else models_url
        status("vLLM OpenAI server", True, detail)
        return True
    except urllib.error.URLError as exc:
        status("vLLM OpenAI server", False, f"not reachable at {models_url}: {exc}")
    except Exception as exc:
        status("vLLM OpenAI server", False, f"not reachable at {models_url}: {exc}")
    return False


def main():
    print(f"SG-Nav setup check: {ROOT}")
    hard_ok = True
    hard_ok &= check_main_env()
    hard_ok &= check_vllm_env()
    hard_ok &= check_assets()

    print()
    check_gpu()
    check_vllm_server()

    print()
    if hard_ok:
        print("Required files and environments are present.")
        print("Start vLLM with: ./run_vllm.sh")
        print("Then run SG-Nav with: ./run_sg_nav.sh --visualize")
        return 0

    print("Some required files or environments are missing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
