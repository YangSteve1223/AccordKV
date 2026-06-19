"""
数据加载器
支持 UAVTrajectory (UAV Delivery) 和 UAV-Flow-Sim 数据集
统一输出格式: (B, T, 6) [x, y, z, vx, vy, vz]

数据来源:
- UAVTrajectory (UAV Delivery): cvmart.net, ~6911条, ATS仿真
- UAV-Flow-Sim: HuggingFace wangxiangyu0814/UAV-Flow, ~30692条, UE仿真
"""

import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, Optional, List
from pathlib import Path


class ChunkedTrajectoryDataset(Dataset):
    """
    通用轨迹数据集 (支持 chunked npz + metadata.json 格式)
    数据组织:
        data_root/
        ├── processed/
        │   ├── trajectories_chunk_*.npz  (多chunk)
        │   ├── metadata.json              (元信息)
        │   └── data_stats.json
        └── raw/
            └── download_status.json
    npz格式:
        positions: (N, max_T, 3)
        velocities: (N, max_T, 3)
        timestamps: (N, max_T)
        masks: (N, max_T)  # 有效数据掩码
    metadata.json: 每条轨迹的元信息
    """
    def __init__(
        self,
        data_root: str,
        hist_len: int = 20,
        pred_len: int = 20,
        dt: float = 0.1,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        min_length: int = 50,
        normalize: bool = True,  # 局部坐标系归一化
    ):
        self.data_root = Path(data_root)
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.dt = dt
        self.normalize = normalize
        self.split = split

        self.positions = []   # list of (T, 3) arrays
        self.velocities = []  # list of (T, 3) arrays
        self.intent_labels = []  # list of int
        self.maneuver_levels = []  # list of int

        self._load_chunks()
        self._split_dataset(train_ratio, val_ratio)
        self._compute_statistics()

    def _load_chunks(self):
        """加载所有npz chunks"""
        processed_dir = self.data_root / 'processed'
        chunk_files = sorted(processed_dir.glob('trajectories_chunk_*.npz'))

        if not chunk_files:
            # 尝试 merged 文件作为快速路径
            merged_file = processed_dir / 'trajectories_merged.npz'
            if merged_file.exists():
                print(f"Loading merged file: {merged_file}")
                merged = np.load(merged_file, allow_pickle=True)
                self.raw_positions = [merged['positions']]
                self.raw_velocities = [merged['velocities']]
                self.raw_masks = [merged['masks']]
                self.all_positions = merged['positions']
                self.all_velocities = merged['velocities']
                self.all_masks = merged['masks']
                # 加载 metadata（如果存在）
                meta_path = processed_dir / 'metadata.json'
                if meta_path.exists():
                    with open(meta_path) as f:
                        self.metadata = json.load(f)
                else:
                    self.metadata = []
                self._process_trajectories()
                return
            raise FileNotFoundError(f"No chunk files in {processed_dir}")

        # 加载metadata
        meta_path = processed_dir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        # 加载npz chunks
        self.raw_positions = []
        self.raw_velocities = []
        self.raw_masks = []

        for chunk_file in chunk_files:
            data = np.load(chunk_file, allow_pickle=True)
            self.raw_positions.append(data['positions'])
            self.raw_velocities.append(data['velocities'])
            self.raw_masks.append(data['masks'])
            print(f"Loaded {chunk_file.name}: {data['positions'].shape[0]} trajectories")

        # 合并所有chunks
        self.all_positions = np.concatenate(self.raw_positions, axis=0)
        self.all_velocities = np.concatenate(self.raw_velocities, axis=0)
        self.all_masks = np.concatenate(self.raw_masks, axis=0)

        self._process_trajectories()

    def _process_trajectories(self):
        """处理轨迹: 滑动窗口切分"""
        stride = 5  # 滑动步长

        for idx in range(len(self.all_positions)):
            pos = self.all_positions[idx]   # (T, 3)
            vel = self.all_velocities[idx]  # (T, 3)
            mask = self.all_masks[idx]      # (T,)

            if mask.sum() < self.hist_len + self.pred_len:
                continue

            # 有效帧索引
            valid_idx = np.where(mask > 0)[0]
            total_len = len(valid_idx)

            # 滑动窗口切分
            for start in range(0, total_len - self.hist_len - self.pred_len + 1, stride):
                hist_idx = valid_idx[start:start + self.hist_len]
                pred_idx = valid_idx[start + self.hist_len:start + self.hist_len + self.pred_len]

                hist_pos = pos[hist_idx]   # (hist_len, 3)
                hist_vel = vel[hist_idx]   # (hist_len, 3)
                pred_pos = pos[pred_idx]   # (pred_len, 3)

                # 局部坐标系归一化
                if self.normalize:
                    origin = hist_pos[0:1, :]
                    hist_pos = hist_pos - origin
                    pred_pos = pred_pos - origin

                # 组合 6D
                hist_6d = np.concatenate([hist_pos, hist_vel], axis=1)  # (hist_len, 6)
                pred_3d = pred_pos[:, :3]  # (pred_len, 3)

                # 意图/机动标签 (从metadata，metadata按全局traj_id索引)
                # all_positions 是所有chunks拼接后的结果，索引idx对应全局traj_id=idx
                if idx < len(self.metadata):
                    meta = self.metadata[idx]
                    # intent 字段可能不存在，用 maneuver_score 和轨迹几何推断
                    if 'intent' in meta:
                        intent = int(meta['intent'])
                    else:
                        intent = self._infer_intent_from_meta(meta)
                    maneuver = int(meta.get('maneuver_level', 1))
                else:
                    intent, maneuver = self._infer_labels(hist_pos, hist_vel)

                self.positions.append(hist_6d.astype(np.float32))
                self.velocities.append(pred_3d.astype(np.float32))
                self.intent_labels.append(intent)
                self.maneuver_levels.append(maneuver)

    def _infer_intent_from_meta(self, meta: dict) -> int:
        """
        从 metadata 的几何特征推断全局意图类别
        meta 含: maneuver_score, mean_gs, max_gs, start_alt 等
        """
        maneuver_score = meta.get('maneuver_score', 0.5)
        max_gs = meta.get('max_gs', 5.0)
        mean_gs = meta.get('mean_gs', 0.0)
        start_alt = meta.get('start_alt', 0.0)

        # 高机动得分 → EVASIVE
        if maneuver_score > 0.7:
            return 6  # EVASIVE
        # 速度变化剧烈 → 可能转弯
        if max_gs > 3.0 and mean_gs > 1.0:
            # 进一步按轨迹曲率方向判断左右
            # 用 heading 估算（metadata 无 heading，用速度方向代理）
            # 简化: 高机动 + 中速 → TURN
            if maneuver_score > 0.4:
                return 2  # TURN_LEFT (默认右手法则)
        # 高度变化明显 → CLIMB/DESCEND
        # 当前数据 start_alt 都约 3m，看 mean_alt 变化
        mean_alt = meta.get('mean_alt', 0.0)
        alt_range = meta.get('max_alt', mean_alt) - meta.get('min_alt', mean_alt)
        if alt_range > 10.0:
            # 上升率判断（需要速度 > 0）
            if mean_gs > 0.5:
                return 4  # CLIMB
            else:
                return 5  # DESCEND
        # 平稳悬停
        if maneuver_score < 0.3 and max_gs < 1.0:
            return 0  # HOVER
        # 默认直线飞行
        return 1  # STRAIGHT

    def _infer_labels(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[int, int]:
        """从轨迹推断意图和机动等级"""
        if len(pos) < 5:
            return 1, 0

        accel = np.diff(vel, axis=0)
        jerk = np.diff(accel, axis=0)
        accel_mag = np.linalg.norm(accel, axis=1).mean()
        jerk_mag = np.linalg.norm(jerk, axis=1).mean()
        vel_std = np.linalg.norm(vel, axis=1).std()

        score = accel_mag * 0.4 + jerk_mag * 0.4 + vel_std * 0.2

        # 机动等级
        if score < 2.0:
            maneuver = 0
        elif score < 5.0:
            maneuver = 1
        else:
            maneuver = 2

        # 意图推断
        heading_changes = np.diff(np.arctan2(vel[:, 1], vel[:, 0]))
        mean_turn = np.abs(heading_changes).mean()
        if mean_turn > 0.05:
            direction = heading_changes.mean()
            intent = 2 if direction < 0 else 3  # TURN_LEFT/RIGHT
        else:
            intent = 1  # STRAIGHT

        # 高度变化
        alt_diff = pos[-1, 2] - pos[0, 2]
        if alt_diff > 2.0:
            intent = 4  # CLIMB
        elif alt_diff < -2.0:
            intent = 5  # DESCEND

        return intent, maneuver

    def _split_dataset(self, train_ratio: float, val_ratio: float):
        n_total = len(self.positions)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if self.split == 'train':
            self.positions = self.positions[:n_train]
            self.velocities = self.velocities[:n_train]
            self.intent_labels = self.intent_labels[:n_train]
            self.maneuver_levels = self.maneuver_levels[:n_train]
        elif self.split == 'val':
            self.positions = self.positions[n_train:n_train + n_val]
            self.velocities = self.velocities[n_train:n_train + n_val]
            self.intent_labels = self.intent_labels[n_train:n_train + n_val]
            self.maneuver_levels = self.maneuver_levels[n_train:n_train + n_val]
        else:  # test
            self.positions = self.positions[n_train + n_val:]
            self.velocities = self.velocities[n_train + n_val:]
            self.intent_labels = self.intent_labels[n_train + n_val:]
            self.maneuver_levels = self.maneuver_levels[n_train + n_val:]

    def _compute_statistics(self):
        self.n_samples = len(self.positions)
        maneuver_arr = np.array(self.maneuver_levels)
        self.maneuver_dist = {
            0: int(np.count_nonzero(maneuver_arr == 0)),
            1: int(np.count_nonzero(maneuver_arr == 1)),
            2: int(np.count_nonzero(maneuver_arr == 2)),
        }

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist = torch.from_numpy(self.positions[idx])      # (hist_len, 6)
        pred = torch.from_numpy(self.velocities[idx])     # (pred_len, 3)
        intent = torch.tensor(self.intent_labels[idx], dtype=torch.long)
        return hist, pred, intent


def get_dataloader(
    dataset_name: str,
    data_root: str,
    hist_len: int = 20,
    pred_len: int = 20,
    batch_size: int = 32,
    num_workers: int = 4,
    split: str = 'train',
    **kwargs
) -> DataLoader:
    """工厂函数"""
    dataset = ChunkedTrajectoryDataset(
        data_root=data_root,
        hist_len=hist_len,
        pred_len=pred_len,
        split=split,
        **kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )


# === 直接从CSV加载 (原始数据格式) ===
def load_uav_delivery_csv(csv_path: str, dt: float = 0.1):
    """直接从CSV加载UAVTrajectory原始数据"""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1,
                       usecols=(2, 3, 4, 5, 6))
    positions = data[:, :3]
    speeds = data[:, 3:4]
    headings = data[:, 4:5]

    velocities = np.diff(positions, axis=0) / dt
    velocities = np.vstack([velocities[0:1], velocities])

    return np.concatenate([positions, velocities], axis=1)


def load_uav_flow_sim_csv(csv_path: str, dt: float = 0.2):
    """直接从CSV加载UAV-Flow-Sim原始数据"""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    positions = data[:, :3]
    velocities = np.diff(positions, axis=0) / dt
    velocities = np.vstack([velocities[0:1], velocities])
    return np.concatenate([positions, velocities], axis=1)
