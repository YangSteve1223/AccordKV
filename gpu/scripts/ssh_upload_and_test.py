#!/usr/bin/env python3
"""
上传 GPU 实验脚本到远程服务器并执行。
"""
import paramiko
import os
import time

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
    """上传单个文件"""
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()
    print(f"  uploaded: {remote_path}")

# 1. 创建远程目录
run(f"mkdir -p {REMOTE_DIR}")

# 2. 上传所有脚本
files = ["gpu_svd_compress.py", "gpu_wire_format.py",
         "gpu_model_loader.py", "gpu_run_exp.py", "__init__.py"]
for f in files:
    lp = os.path.join(LOCAL_DIR, f)
    rp = f"{REMOTE_DIR}/{f}"
    if os.path.exists(lp):
        put_file(lp, rp)
    else:
        print(f"  MISSING: {lp}")

print("\n=== 验证上传 ===")
out, err = run(f"ls -la {REMOTE_DIR}/")

print(out)

# 3. 语法检查
print("\n=== 语法检查 ===")
for f in files:
    if f == "__init__.py":
        continue
    out, err = run(f"{PYBIN} -m py_compile {REMOTE_DIR}/{f}")
    if out or err:
        print(f"[{f}] ERROR: {err}")
    else:
        print(f"[{f}] OK")

# 4. 试运行 SVD 测试（不加载模型，纯算法）
print("\n=== SVD 算法测试 ===")
test_script = f"{REMOTE_DIR}/gpu_svd_compress.py"
out, err = run(f"{PYBIN} {test_script}")
print(out[-2000:] if len(out) > 2000 else out)
if err:
    print("STDERR:", err[-500:])

client.close()
