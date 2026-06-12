#!/usr/bin/env python3
import paramiko
import os

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    "CHANGE_ME_HOST", port=CHANGE_ME_PORT,
    username="root", password="CHANGE_ME_PASSWORD",
    timeout=30, allow_agent=False, look_for_keys=False
)

def run(cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return (stdout.read().decode().strip() or stderr.read().decode().strip())

# Find conda and python
print("=== conda ===")
print(run("which conda && conda --version && ls /root/miniconda3/bin/python* 2>/dev/null | head -10"))
print(run("conda info --envs"))

PYBIN = "/root/miniconda3/bin/python"
cmds = [
    ("python version", f"{PYBIN} --version"),
    ("transformers", f"{PYBIN} -c 'import transformers; print(transformers.__version__)'"),
    ("torch", f"{PYBIN} -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"),
    ("torch cuda", f"{PYBIN} -c 'import torch; print(torch.cuda.get_device_name(0))'"),
    ("vllm", f"{PYBIN} -c 'import vllm; print(vllm.__version__)'"),
    ("gemma model files", "ls /root/autodl-tmp/gemma-2-9b-it/*.safetensors 2>/dev/null | wc -l"),
    ("mistral model files", "ls /root/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/*.safetensors 2>/dev/null | head -5"),
    ("gpu mem", "nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader"),
    ("gpu compute", "nvidia-smi --query-gpu=compute_cap --format=csv,noheader"),
]

print("\n=== environment ===")
for name, cmd in cmds:
    out = run(cmd)
    print(f"[{name}] {out}")

client.close()
