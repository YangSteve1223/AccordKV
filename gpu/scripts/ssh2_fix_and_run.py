#!/usr/bin/env python3
"""上传修复后的 gpu_svd_compress.py 到服务器，跑 SVD 自测，然后跑真实实验"""
import paramiko, time, os, json
from pathlib import Path

HOST = "CHANGE_ME_HOST"
PORT = CHANGE_ME_PORT
USER = "root"
PASS = "CHANGE_ME_PASSWORD"
PY   = "/root/miniconda3/bin/python"

REMOTE_DIR  = "/root/accord-kv/gpu"
REMOTE_RES  = "/root/accord-kv/gpu_results"
REMOTE_EXP_LOG = "/root/accord-kv/exp.log"

LOCAL_FIXED = "/app/data/所有对话/主对话/_staging/accord-kv/gpu_svd_compress_fixed.py"
LOCAL_LOADER = "/app/data/所有对话/主对话/_staging/accord-kv/gpu_model_loader.py"

# ── 1. SSH ──
print("Connecting...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = ssh.open_sftp()
print("Connected ✅")

# ── 2. 上传修复后的压缩脚本 ──
print("\n上传 gpu_svd_compress_fixed.py → /root/accord-kv/gpu/gpu_svd_compress.py ...")
sftp.put(LOCAL_FIXED, f"{REMOTE_DIR}/gpu_svd_compress.py")
print("Upload done ✅")

# ── 3. 跑 SVD 自测 ──
print("\n跑 SVD 自测...")
stdin, stdout, stderr = ssh.exec_command(f"{PY} {REMOTE_DIR}/gpu_svd_compress.py 2>&1")
out = stdout.read().decode()
err = stderr.read().decode()
print(out)
if err:
    print("STDERR:", err)

# ── 4. 跑真实实验 (Mistral seq=512 rank=8) ──
print("\n" + "="*60)
print("启动 Mistral-7B-Instruct GPU 实验...")
print("="*60)

exp_script = f"""#!/bin/bash
{PY} -c "
import sys; sys.path.insert(0, '{REMOTE_DIR}')
import torch, json, os
from gpu_model_loader import load_model_and_get_hook
from gpu_svd_compress import compress_kv_full, decompress_kv, measure_reconstruction_error

os.makedirs('{REMOTE_RES}', exist_ok=True)

# 加载模型
print('加载 Mistral...')
model, device, tok = load_model_and_get_hook('mistral')
print(f'模型加载完成 device={{device}}')

# KV hook
class KVHolder:
    def __init__(self): self.k_cache, self.v_cache = [], []
    def hook_fn(self, m, inp, out):
        # out: (logits, (k_state, v_state)) for HF
        kv = out[1] if isinstance(out, tuple) else out
        if isinstance(kv, tuple):
            k, v = kv[0].detach(), kv[1].detach()
        else:
            k = v = kv.detach()
        self.k_cache.append(k.float()); self.v_cache.append(v.float())
        return out

holder = KVHolder()
register_hook = None

# 尝试注册 hook
try:
    from transformers import AutoModelForCausalLM
    # 找 attn 层注册
    for name, module in model.named_modules():
        if 'attn' in name.lower() or 'k_proj' in name:
            module.register_forward_hook(holder.hook_fn)
            print(f'Registered hook on: {{name}}')
            break
except Exception as e:
    print(f'Hook 注册失败: {{e}}')

# 推理
seq_len = 512
input_ids = torch.randint(0, tok.vocab_size, (1, seq_len), device=device)
print(f'Run inference seq={{seq_len}}...')

with torch.no_grad():
    _ = model(input_ids)

# 从 holder 拿 KV
if holder.k_cache:
    k_full = torch.cat([k.unsqueeze(0) for k in holder.k_cache], dim=2)  # [B,H,T,D]
    v_full = torch.cat([v.unsqueeze(0) for v in holder.v_cache], dim=2)
    # squeeze batch
    k_full = k_full[0]; v_full = v_full[0]
    print(f'KV shape: K={{k_full.shape}} V={{v_full.shape}}')
else:
    print('ERROR: KV cache 为空，hook 未捕获到数据')
    sys.exit(1)

# SVD 压缩/解压
rank = 8
comp = compress_kv_full(k_full, v_full, rank=rank, quantize=True, int4=True)
k_r, v_r = decompress_kv(comp, quantize=True, int4=True)
err = measure_reconstruction_error(k_full, v_full, k_r, v_r)

print('=== 结果 ===')
print(json.dumps({{'rank': rank, 'seq_len': seq_len, 'model': 'Mistral-7B-Instruct-v0.3', **err}}))
with open('{REMOTE_RES}/mistral_result.json', 'w') as f:
    json.dump({{'rank': rank, 'seq_len': seq_len, 'model': 'Mistral-7B-Instruct-v0.3', **err}}, f, indent=2)
print('结果已保存到 {REMOTE_RES}/mistral_result.json')
" 2>&1 | tee {REMOTE_EXP_LOG}
"""
    
# 写实验脚本
sftp.write(f"{REMOTE_DIR}/run_mistral_exp.py", exp_script.lstrip())
print("实验脚本已写入 ✅")

# 后台跑
print("启动实验（后台 nohup）...")
chan = ssh.exec_command(f"cd {REMOTE_DIR} && nohup {PY} run_mistral_exp.py > {REMOTE_EXP_LOG} 2>&1 &\necho 'PID=$!'")
pid_out = chan[1].read().decode()
print(f"后台 PID: {pid_out}")
print(f"日志路径: {REMOTE_EXP_LOG}")

# 等 30 秒看初步输出
print("\n等待 30 秒看日志...")
time.sleep(30)

# 读日志
chan = ssh.exec_command(f"tail -50 {REMOTE_EXP_LOG}")
log = chan[1].read().decode()
print("\n=== 当前日志 ===")
print(log)

sftp.close()
ssh.close()
print("\n连接已关闭，实验在后台运行中")
