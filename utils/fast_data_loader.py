#!/usr/bin/env python3
"""
快速 DataLoader：直接从预生成的 windows_*.npz chunk 文件加载
Dataset init 时无 Python 循环、无 np.concatenate，纯直接读取
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Literal


class FastWindowDataset(Dataset):
    """
    直接加载预生成的 windows_cache 文件（可以是多个 chunk）
    
    npz 文件结构：
        hist_windows  : (N, hist_len, 6)  [pos+vel, float32]
        pred_windows  : (N, pred_len, 3)  [pos delta, float32]
        intent_labels  : (N,)  [int32]
        maneuver_levels : (N,)  [int32]
    """
    def __init__(
        self,
        data_root: str,
        split: Literal['train', 'val', 'test'] = 'train',
    ):
        self.data_root = Path(data_root)
        self.processed = self.data_root  # 数据直接在 data/NPZDATA 下，无 processed 子目录
        self.split = split

        # 递归搜索子目录中的 chunk 文件
        self.chunk_files = sorted(self.processed.rglob(f'windows_{split}_chunk*.npz'))

        if not self.chunk_files:
            raise FileNotFoundError(f"No windows_{split}_chunk*.npz in {self.processed}")

        # 加载每个 chunk 的索引范围（跳过损坏文件）
        self.chunk_offsets = [0]
        self.n_per_chunk = []
        self.valid_chunk_files = []
        for f in self.chunk_files:
            try:
                data = np.load(f)
                n = len(data['hist'])  # key 一致：hist, pred, intent, maneuver
                self.n_per_chunk.append(n)
                self.chunk_offsets.append(self.chunk_offsets[-1] + n)
                self.valid_chunk_files.append(f)
                del data
            except Exception as e:
                print(f"[FastWindowDataset] 跳过损坏文件 {f}: {e}")

        if not self.valid_chunk_files:
            raise FileNotFoundError(f"No valid chunks for {split} in {self.processed}")
        self.chunk_files = self.valid_chunk_files

        self.n_samples = self.chunk_offsets[-1]

        # 加载元信息
        import json
        meta_path = self.processed / 'windows_meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """直接随机读取，无需预读"""
        # 定位到哪个 chunk
        for ci, (start, n) in enumerate(zip(self.chunk_offsets[:-1], self.n_per_chunk)):
            if idx < start + n:
                local_idx = idx - start
                break
        else:
            local_idx = 0

        data = np.load(self.chunk_files[ci])
        hist = torch.from_numpy(data['hist'][local_idx])
        pred = torch.from_numpy(data['pred'][local_idx])
        intent = torch.tensor(data['intent'][local_idx], dtype=torch.long)
        del data

        return hist, pred, intent


def get_dataloader(data_root='/home/featurize/data/NPZDATA', split='train', batch_size=32, num_workers=2, shuffle=None):
    """工厂函数"""
    if shuffle is None:
        shuffle = (split == 'train')
    dataset = FastWindowDataset(data_root=data_root, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    return loader


if __name__ == '__main__':
    import time
    data_root = '/home/featurize/data/NPZDATA'

    for split in ['train', 'val', 'test']:
        t0 = time.time()
        ds = FastWindowDataset(data_root, split=split)
        print(f"{split}: {len(ds)} 样本, init={time.time()-t0:.1f}s")

    # 测试加载速度
    ds = FastWindowDataset(data_root, split='train')
    t0 = time.time()
    for i in range(100):
        _ = ds[i % len(ds)]
    print(f"100 次随机读取: {time.time()-t0:.2f}s")

    # 测试 DataLoader
    loader = get_dataloader(data_root, split='train', batch_size=32, num_workers=0)
    t0 = time.time()
    for batch in loader:
        print(f"  batch: hist={batch[0].shape}, pred={batch[1].shape}, intent={batch[2].shape}")
        break
    print(f"第一个 batch: {time.time()-t0:.2f}s")
