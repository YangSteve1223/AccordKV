#!/usr/bin/env python3
import paramiko, time

HOST, PORT = "CHANGE_ME_HOST", CHANGE_ME_PORT
USER, PWD = "root", "CHANGE_ME_PASSWORD"
PYBIN = "/root/miniconda3/bin/python"
REMOTE = "/root/accord-kv/gpu"

# 读本地修复版
with open("/app/data/所有对话/主对话/simulation/gpu/gpu_svd_compress.py") as f:
    SVD = f.read()
with open("/app/data/所有对话/主对话/simulation/gpu/gpu_model_loader.py") as f:
    ML = f.read()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)

def run(cmd, timeout=120):
    s,o,i = client.exec_command(cmd, timeout=timeout)
    return o.read().decode(), i.read().decode()

# SFTP 写文件
sftp = client.open_sftp()
sftp.file(f"{REMOTE}/gpu_svd_compress.py","w").write(SVD)
sftp.file(f"{REMOTE}/gpu_model_loader.py","w").write(ML)
sftp.close()
print("Files written")

# 停止旧进程
run("kill $(ps aux|grep gpu_run_exp|grep -v grep|awk '{print $2}') 2>/dev/null; echo ok")
run("mkdir -p /root/accord-kv/gpu_results")

# 语法检查
for fn in ["gpu_svd_compress.py","gpu_model_loader.py"]:
    o,e = run(f"{PYBIN} -m py_compile {REMOTE}/{fn}")
    status = "OK" if not e.strip() else e.strip()[:100]
    print(f"  [{fn}] {status}")

# SVD 自测
print("\n=== SVD测试 ===")
o,e = run(f"cd {REMOTE} && {PYBIN} gpu_svd_compress.py")
print(o)
if "ALL TESTS PASSED" not in o:
    print("SVD FAIL:", o[-500:], e[-300:])
else:
    print("SVD PASS!")

client.close()
