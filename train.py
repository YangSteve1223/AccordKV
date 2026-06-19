"""
训练脚本
支持多数据集、多阶段训练、断点续训、模型权重导出
"""

import os
import sys
import argparse
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from emam_model import TrajectoryPredictor
from utils.fast_data_loader import FastWindowDataset, get_dataloader
from utils.metrics import (
    full_evaluation, compute_displacement_error,
    compute_distance_accuracy, compute_direction_accuracy,
    maneuver_classification
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train EMam-SE trajectory prediction')
    # 数据
    parser.add_argument('--dataset', type=str, default='uav_delivery',
                        choices=['uav_delivery', 'uav_flow_sim'])
    parser.add_argument('--data_root', type=str,
                        default='/home/featurize/data/NPZDATA')
    parser.add_argument('--hist_len', type=int, default=20)
    parser.add_argument('--pred_len', type=int, default=20)
    # 模型
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--emam_n_layers', type=int, default=2)
    parser.add_argument('--use_trigger', type=int, default=1)
    # 训练
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'])
    parser.add_argument('--warmup_epochs', type=int, default=5)
    # 损失权重
    parser.add_argument('--loss_disp_weight', type=float, default=1.0)
    parser.add_argument('--loss_intent_weight', type=float, default=0.1)
    parser.add_argument('--loss_unc_weight', type=float, default=0.05)
    # 输出
    parser.add_argument('--exp_name', type=str, default='emam_se_default')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=50)
    # 断点续训
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    # 硬件
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def build_model(args):
    model = TrajectoryPredictor(
        input_dim=6,
        history_len=args.hist_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        emam_n_layers=args.emam_n_layers,
        use_trigger=bool(args.use_trigger),
        loss_weights={
            'displacement': args.loss_disp_weight,
            'intent': args.loss_intent_weight,
            'uncertainty': args.loss_unc_weight
        }
    )
    return model


def build_optimizer(model, args):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    return optimizer


def build_scheduler(optimizer, args, total_steps):
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=len(optimizer.param_groups[0]['params']) * 10, gamma=0.5
        )
    else:
        scheduler = None
    return scheduler


def train_epoch(model, train_loader, optimizer, device, epoch, args, writer=None):
    model.train()
    total_loss = 0
    total_samples = 0
    loss_breakdown = {'displacement': 0, 'intent': 0, 'uncertainty': 0}

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (hist, pred, intent_labels) in enumerate(pbar):
        hist = hist.to(device)        # (B, hist_len, 6)
        pred = pred.to(device)        # (B, pred_len, 3)
        intent_labels = intent_labels.to(device)  # (B,)

        # Forward
        out = model(hist, intent_labels=intent_labels, return_all=False)
        predictions = out['predictions']       # (B, pred_len, 3)
        uncertainty = out['uncertainty']     # (B, pred_len, 3)
        intent_logits = out['intent_logits']  # (B, num_classes)

        # 计算位移真值 (历史最后位置 → 未来位置 的增量)
        targets = pred[..., :3]  # (B, pred_len, 3)

        # 损失
        losses = model.compute_loss(
            predictions, uncertainty, targets,
            intent_logits, intent_labels
        )
        loss = losses['total_loss']

        # 反向
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        B = hist.shape[0]
        total_loss += loss.item() * B
        total_samples += B
        for k in loss_breakdown:
            loss_breakdown[k] += losses[f'loss_{k}'].item() * B

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_samples
    return {k: v / total_samples for k, v in loss_breakdown.items()}, avg_loss


@torch.no_grad()
def validate(model, val_loader, device, epoch, args):
    model.eval()
    all_predictions = []
    all_targets = []
    all_intents_pred = []
    all_intents_true = []

    for hist, pred, intent_labels in tqdm(val_loader, desc='Validating'):
        hist = hist.to(device)
        pred = pred.to(device)
        intent_labels = intent_labels.to(device)

        out = model(hist, force_predict=True)
        predictions = out['predictions']
        intent_logits = out['intent_logits']

        all_predictions.append(predictions.cpu())
        all_targets.append(pred[..., :3].cpu())
        all_intents_pred.append(intent_logits.argmax(dim=-1).cpu())
        all_intents_true.append(intent_labels.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_intents_pred = torch.cat(all_intents_pred, dim=0)
    all_intents_true = torch.cat(all_intents_true, dim=0)

    # 评估
    results = full_evaluation(all_predictions, all_targets)
    intent_acc = (all_intents_pred == all_intents_true).float().mean().item()
    results['intent_accuracy'] = intent_acc

    return results


def save_checkpoint(model, optimizer, scheduler, epoch, results, args, is_best=False):
    """保存模型检查点"""
    ckpt_dir = Path(args.checkpoint_dir) / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results,
        'args': vars(args)
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()

    # 最新检查点
    latest_path = ckpt_dir / 'latest.pth'
    torch.save(ckpt, latest_path)

    # 定期保存
    if epoch % args.save_every == 0:
        torch.save(ckpt, ckpt_dir / f'epoch_{epoch}.pth')

    # 最佳模型
    if is_best:
        torch.save(ckpt, ckpt_dir / 'best.pth')

    # 保存config
    config_path = ckpt_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f)

    return str(latest_path)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 数据
    print(f"Loading dataset: {args.dataset}")
    train_loader = get_dataloader(
        args.data_root,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        args.data_root,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # 模型
    model = build_model(args).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 优化器
    optimizer = build_optimizer(model, args)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_scheduler(optimizer, args, total_steps)

    # 断点续训
    start_epoch = 0
    best_metric = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if 'results' in ckpt and 'RMSE' in ckpt['results']:
            best_metric = ckpt['results']['RMSE']

    # TensorBoard
    writer = SummaryWriter(log_dir=f'./runs/{args.exp_name}')

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # Warmup
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        loss_breakdown, train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, args, writer
        )

        # 验证
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_results = validate(model, val_loader, device, epoch, args)

            # 判断是否最优
            is_best = val_results['RMSE'] < best_metric
            if is_best:
                best_metric = val_results['RMSE']

            print(f"\nEpoch {epoch} Val Results:")
            print(f"  RMSE: {val_results['RMSE']:.4f}")
            print(f"  Distance Accuracy: {val_results['Distance_Accuracy']:.4f}")
            print(f"  Direction Accuracy: {val_results['Direction_Accuracy']:.4f}")
            print(f"  Intent Accuracy: {val_results.get('intent_accuracy', 0):.4f}")
            print(f"  Mean Jerk: {val_results['Mean_Jerk']:.4f}")

            # TensorBoard
            for k, v in val_results.items():
                writer.add_scalar(f'val/{k}', v, epoch)

            ckpt_path = save_checkpoint(model, optimizer, scheduler, epoch, val_results, args, is_best)
            print(f"Checkpoint: {ckpt_path}")

        # TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        for k, v in loss_breakdown.items():
            writer.add_scalar(f'train/loss_{k}', v, epoch)

        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")
        print(f"  Displacement: {loss_breakdown['displacement']:.4f}")
        print(f"  Intent: {loss_breakdown['intent']:.4f}")
        print(f"  Uncertainty: {loss_breakdown['uncertainty']:.4f}")

    writer.close()
    print(f"\nTraining done. Best RMSE: {best_metric:.4f}")


if __name__ == '__main__':
    main()
