#!/usr/bin/env python3
"""上传修复脚本 + SVD 自测 + 后台启动 Mistral 实验"""
import paramiko, time

HOST = "CHANGE_ME_HOST"
PORT = CHANGE_ME_PORT
USER = "root"
PASS = "CHANGE_ME_PASSWORD"
PY   = "/root/miniconda3/bin/python"
REMOTE_DIR = "/root/accord-kv/gpu"
REMOTE_RES = "/root/accord-kv/gpu_results"
REMOTE_LOG = "/root/accord-kv/exp.log"

LOCAL_DIR = "/app/data/所有对话/主对话/_staging/accord-kv"

FILES = {
    "gpu_svd_compress.py":     "/app/data/所有对话/主对话/_staging/accord-kv/gpu_svd_compress_fixed.py",
    "gpu_model_loader.py":     "/app/data/所有对话/主对话/gpu_model_loader.py",
    "gpu_run_mistral_exp.py":  "/app/data/所有对话/主对话/_staging/accord-kv/gpu_run_mistral_exp.py",
}

def cmd(ssh, c):
    stdin, stdout, stderr = ssh.exec_command(c)
    return stdout.read().decode(), stderr.read().decode()

print("连接服务器...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = ssh.open_sftp()
print("✅ 已连接\n")

# ── 1. 上传文件 ──
for fname, lpath in FILES.items():
    print(f"上传 {fname}...")
    sftp.put(lpath, f"{REMOTE_DIR}/{fname}")
print("✅ 全部上传完成\n")

# ── 2. SVD 自测 ──
print("=" * 60)
print("SVD 自测")
print("=" * 60)
out, err = cmd(ssh, f"{PY} {REMOTE_DIR}/gpu_svd_compress.py 2>&1")
print(out)
if err: print("STDERR:", err)

# 判断是否 PASS
svd_pass = "✅ PASS" in out and "❌ FAIL" not in out
if not svd_pass:
    print("\n❌ SVD 自测未通过，先不启动实验")
    sftp.close(); ssh.close()
    exit(1)
print("\n✅ SVD 自测通过，启动真实实验\n")

# ── 3. 清空旧日志，启动实验 ──
cmd(ssh, f"> {REMOTE_LOG}")
cmd(ssh, f"cd {REMOTE_DIR} && nohup {PY} gpu_run_mistral_exp.py > {REMOTE_LOG} 2>&1 &\necho 'PID=$!'")
print("✅ 实验已在后台启动")

# ── 4. 等 60 秒，看早期日志 ──
print("\n等待 60 秒看早期输出...")
time.sleep(60)

out, _ = cmd(ssh, f"tail -60 {REMOTE_LOG}")
print("\n=== 日志片段 ===")
print(out)

sftp.close()
ssh.close()
print("\n连接已关闭，实验继续在后台运行")
