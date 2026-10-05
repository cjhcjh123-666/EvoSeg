"""Bounded, checkpointed native SaSaSa2VA fine-tune on public training videos."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .native_data import PublicMevisNative
from .native_model import NativeFineTune, load_release, prepare_adaptation, restore_adaptation
from .prepare_foundation import atomic_json


def save_checkpoint(model, optimizer, args, step, path, split, epoch, next_batch, micro):
    state = {name: parameter.detach().cpu().clone()
             for name, parameter in model.named_parameters() if parameter.requires_grad}
    checkpoint = {'format': 'native_sasasa2va_adaptation_v1', 'trainable_state': state,
                  'optimizer': optimizer.state_dict(), 'step': step,
                  'lora_rank': args.lora_rank, 'arguments': vars(args),
                  'split': split, 'epoch': epoch, 'next_batch': next_batch, 'micro': micro,
                  'foundation_revision': json.loads(Path(args.assets).read_text())['revision']}
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def run(args):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_file = output_dir / 'TRAINING_STATUS.json'
    state = {'status': 'LOADING_RELEASE', 'training_started': False,
             'world_size': world, 'started_at_unix': time.time(), 'pid': os.getpid()}
    if rank == 0:
        atomic_json(status_file, state)
    model, tokenizer, loading = load_release(args.model_dir, args.assets, device)
    resumed = None
    if args.resume:
        resumed = restore_adaptation(model, args.resume)
        if resumed['foundation_revision'] != loading['revision'] or resumed['lora_rank'] != args.lora_rank:
            raise RuntimeError('resume foundation or adapter configuration differs')
    else:
        prepare_adaptation(model, args.lora_rank)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(.9, .999), weight_decay=.05)
    if resumed:
        optimizer.load_state_dict(resumed['optimizer'])
    dataset = PublicMevisNative(args.train_root, tokenizer, model.conv_template, seed=args.seed)
    split = {'train_videos': dataset.train_ids, 'held_out_videos': dataset.held_out_ids,
             'held_out_scope': 'excluded from this fine-tune; may be seen by published pretraining'}
    if resumed and resumed['split'] != split:
        raise RuntimeError('resume dataset split differs')
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=0)
    network = NativeFineTune(model).train()
    if world > 1:
        network = DistributedDataParallel(network, device_ids=[local_rank], find_unused_parameters=True,
                                           broadcast_buffers=False)
    if rank == 0:
        atomic_json(output_dir / 'SPLIT.json', split)
        atomic_json(output_dir / 'LOADING.json', loading)
        atomic_json(output_dir / 'CONFIG.json', vars(args))
    optimizer.zero_grad(set_to_none=True)
    started, step, micro, epoch = time.time(), 0, 0, 0
    resume_batch = 0
    if resumed:
        step, micro, epoch, resume_batch = (resumed[key] for key in ('step', 'micro', 'epoch', 'next_batch'))
    initial_step = step
    stop_update = min(args.max_updates, args.stop_updates or args.max_updates)
    stop = torch.tensor(0, device=device)
    status_losses = {}
    while step < args.max_updates:
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        for batch_index, sample in enumerate(loader):
            if batch_index < resume_batch:
                continue
            sync = (micro + 1) % args.accumulation == 0
            context = network.no_sync() if world > 1 and not sync else nullcontext()
            with context:
                # Disable the cast cache: frozen teacher weights are large, and
                # no inference-era casts can be reused for a trainable graph.
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    loss, losses = network(sample)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'non-finite loss at micro-step {micro}')
                (loss / args.accumulation).backward()
            micro += 1
            if not sync:
                continue
            # Every rank participates in timeout decisions at a synchronized
            # optimizer boundary. Never leave one rank waiting for another.
            if step == 0:
                groups = {'lora': [], 'prompt': [], 'decoder': []}
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        key = ('lora' if 'lora_' in name else 'prompt' if 'text_hidden_fcs' in name else 'decoder')
                        groups[key].append(float(parameter.grad.detach().abs().sum()))
                if any(not values or not math.isfinite(sum(values)) or sum(values) == 0
                       for values in groups.values()):
                    raise RuntimeError(f'initial native gradient contract failed: { {k:sum(v) for k,v in groups.items()} }')
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            warmup = max(1, round(args.max_updates * .05))
            multiplier = min(1., (step + 1) / warmup) * .5 * (
                1 + math.cos(math.pi * max(0, step - warmup) / max(1, args.max_updates - warmup)))
            for group in optimizer.param_groups:
                group['lr'] = args.lr * multiplier
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            # Aggregate log scalars, not only rank-zero's random video.
            scalars = torch.stack([loss.detach(), *losses.values(), grad_norm.detach().float()])
            if world > 1:
                dist.all_reduce(scalars)
                scalars /= world
            status_losses = dict(zip(['total', *losses, 'grad_norm'], scalars.cpu().tolist()))
            now = time.time()
            stop = torch.tensor(int(now >= args.stop_at or step >= stop_update), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            elapsed = now - started
            state.update(status='TRAINING', training_started=True, step=step, epoch=epoch,
                         max_updates=args.max_updates, losses=status_losses,
                         seconds_per_update=elapsed / (step - initial_step), updated_at_unix=now,
                         gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            if rank == 0:
                atomic_json(status_file, state)
                print(json.dumps(state), flush=True)
            if step % args.save_every == 0 or stop.item():
                checkpoint_path = output_dir / f'checkpoint_{step:05d}.pth'
                if rank == 0:
                    save_checkpoint(model, optimizer, args, step, checkpoint_path, split,
                                    epoch, batch_index + 1, micro)
                    atomic_json(output_dir / 'LATEST.json', {'checkpoint': str(checkpoint_path), 'step': step})
                if world > 1:
                    dist.barrier()
            if stop.item():
                break
        if stop.item():
            break
        resume_batch = 0
        epoch += 1
    state.update(status='FIRST_RUN_COMPLETE', step=step, finished_at_unix=time.time(),
                 stop_reason=('update_limit' if step >= args.max_updates else
                              'pilot_gate' if step >= stop_update else 'evaluation_time_reserved'),
                 evaluated=False)
    if rank == 0:
        atomic_json(status_file, state)
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--train-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-updates', type=int, default=200)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--lora-rank', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--stop-at', type=float, required=True, help='Unix deadline; checkpoint then evaluate')
    parser.add_argument('--stop-updates', type=int, help='Temporary pilot gate under the full-run LR schedule')
    parser.add_argument('--resume', help='Continue a trusted local native adaptation checkpoint')
    args = parser.parse_args()
    if min(args.max_updates, args.accumulation, args.lora_rank, args.save_every) < 1:
        parser.error('update, accumulation, LoRA and save counts must be positive')
    if time.time() >= args.stop_at:
        parser.error('training window already expired')
    if args.stop_updates is not None and args.stop_updates < 1:
        parser.error('pilot updates must be positive')
    try:
        run(args)
    except Exception as error:
        if int(os.environ.get('RANK', 0)) == 0:
            atomic_json(Path(args.output) / 'FAILURE.json', {'type': type(error).__name__,
                'error': str(error), 'updated_at_unix': time.time(), 'success': False})
        raise


if __name__ == '__main__':
    main()
