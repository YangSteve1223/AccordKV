#!/usr/bin/env python3
"""
上传修复后的脚本 + 跑实验（Mistral 优先，7B 模型轻量）。
"""
import paramiko
import os
import time
import sys

HOST = "CHANGE_ME_HOST"
PORT = CHANGE_ME_PORT
USER = "root"
PWD  = "CHANGE_ME_PASSWORD"
PYBIN = "/root/miniconda3/bin/python"

LOCAL_DIR = "/app/data/所有对话/主对话/simulation/gpu"
REMOTE_DIR = "/root/accord-kv/gpu"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PWD,
               timeout=30, allow_agent=False, look_for_keys=False)

def run(cmd, timeout=60):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(), stderr.read().decode()

def put_file(local_path, remote_path):
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()

# 1. 上传修复后的文件
files = ["gpu_svd_compress.py", "gpu_wire_format.py",
         "gpu_model_loader.py", "gpu_run_exp.py"]
print("=== 上传脚本 ===")
for f in files:
    lp = os.path.join(LOCAL_DIR, f)
    rp = f"{REMOTE_DIR}/{f}"
    put_file(lp, rp)
    print(f"  uploaded: {f}")

# 2. 语法验证
print("\n=== 语法检查 ===")
for f in files:
    out, err = run(f"{PYBIN} -m py_compile {REMOTE_DIR}/{f}")
    status = "OK" if not err.strip() else f"ERROR: {err.strip()[:200]}"
    print(f"  [{f}] {status}")

# 3. SVD 算法测试
print("\n=== SVD 算法测试 ===")
out, err = run(f"cd {REMOTE_DIR} && {PYBIN} gpu_svd_compress.py")
print(out[-1500:] if len(out) > 1500 else out)
if "All tests passed" in out:
    print("✅ SVD 算法验证通过")
elif err:
    print("STDERR:", err[-500:])

# 4. 跑 Mistral 实验（--no-vllm，seq=512 先测）
print("\n=== 运行 Mistral 实验 ===")
# 先确保输出目录存在
run(f"mkdir -p /root/accord-kv/gpu_results")
cmd = (
    f"cd {REMOTE_DIR} && {PYBIN} gpu_run_exp.py "
    f"--model mistralai/Mistral-7B-Instruct-v0.3 "
    f"--seq-len 512 "
    f"--rank 8 "
    f"--no-vllm "
    f"--output-dir /root/accord-kv/gpu_results "
    f"2>&1"
)
print(f"Command: {cmd}")

# 启动实验（非阻塞）
transport = client.get_transport()
channel = transport.open_session()
channel.exec_command(cmd)
print(f"Experiment started (pid check via channel)")

# 等待一段时间后检查输出
time.sleep(30)
# 实时读取输出
while channel.recv_ready():
    data = channel.recv(4096).decode()
    print(data, end="")
    sys.stdout.flush()

if channel.exit_status_ready():
    print(f"\n=== 实验结束，状态码: {channel.exit_status()} ===")
    # 读取剩余输出
    while channel.recv_ready():
        print(channel.recv(4096).decode(), end="")
else:
    print("\n实验进行中，通道保持打开...")

client.close()
