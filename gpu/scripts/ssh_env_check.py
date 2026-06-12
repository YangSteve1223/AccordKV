#!/usr/bin/env python3
import paramiko
import sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    "CHANGE_ME_HOST", port=CHANGE_ME_PORT,
    username="root", password="CHANGE_ME_PASSWORD",
    timeout=30, allow_agent=False, look_for_keys=False
)

def run(cmd):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
    return (stdout.read().decode().strip() or stderr.read().decode().strip())

cmds = [
    ("transformers", "python3 -c 'import transformers; print(transformers.__version__)'"),
    ("torch", "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"),
    ("torch cuda name", "python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no\")'"),
    ("vllm", "python3 -c 'import vllm; print(vllm.__version__)'"),
    ("gemma dir", "ls /root/autodl-tmp/gemma-2-9b-it/ 2>/dev/null | head -5"),
    ("mistral dir", "ls /root/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/ 2>/dev/null | head -5"),
    ("gpu mem", "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"),
    ("disk /root", "df -h /root | tail -1"),
    ("disk /tmp", "df -h /tmp | tail -1"),
]

for name, cmd in cmds:
    out = run(cmd)
    print(f"[{name}] {out}")

client.close()
