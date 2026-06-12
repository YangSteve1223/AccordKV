#!/usr/bin/env python3
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    "CHANGE_ME_HOST", port=CHANGE_ME_PORT,
    username="root", password="CHANGE_ME_PASSWORD",
    timeout=30, allow_agent=False, look_for_keys=False
)

def run(cmd, timeout=60):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return (stdout.read().decode().strip() or stderr.read().decode().strip())

# 更全面扫模型
print("=== 扫描 Mistral ===")
print(run("find /root -type d -name '*mistral*' 2>/dev/null | grep -v '.pyc' | grep -v 'conda'"))
print(run("find /home -type d -name '*mistral*' 2>/dev/null | grep -v '.pyc'"))
print(run("find /data -type d -name '*mistral*' 2>/dev/null | grep -v '.pyc' 2>/dev/null || echo 'no /data'"))
print(run("find /mnt -type d -name '*mistral*' 2>/dev/null | grep -v '.pyc' 2>/dev/null || echo 'no /mnt'"))
print(run("find /autodl-tmp -type d 2>/dev/null | head -20"))
print(run("ls -la /root/autodl-tmp/ 2>/dev/null"))
print(run("ls /root/.cache/huggingface/hub/ 2>/dev/null"))

print("\n=== Gemma ===")
print(run("ls /root/autodl-tmp/gemma-2-9b-it/"))

print("\n=== 模型config扫描 ===")
print(run("find /root -name 'config.json' -path '*/Mistral*' 2>/dev/null | head -5"))

client.close()
