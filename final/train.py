"""Train / test semantic segmentation models on ISPRS Vaihingen (or LoveDA Urban).

Examples:
    python train.py --model UNetformer
    python train.py --model ResNet
    python train.py --model ViT
    python train.py --model ViT --mode Test --weights results/ViT_Vaihingen_best.pth

Everything printed during a run is also written to log/<model>_<dataset>_<time>.log.
Weights go to results/: <model>_<dataset>_best.pth (best validation mIoU) and
<model>_<dataset>_final.pth (after the last epoch); one row per run is appended
to results/summary.csv.
"""
import argparse
import csv
import datetime
import json
import os
import random
import sys
import time
import traceback

import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from tqdm import tqdm
from torch.autograd import Variable

from model.UNetFormer import UNetFormer
from model.ResNet import ResNetUNet
from model.ViT import ViTSeg

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, 'results')
LOG_DIR = os.path.join(ROOT, 'log')
MODELS = ['UNetformer', 'ResNet', 'ViT']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='UNetformer', choices=MODELS)
    p.add_argument('--mode', default='Train', choices=['Train', 'Test'])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--milestones', type=int, nargs='+', default=[25, 35, 45])
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--val-interval', type=int, default=500, help='validate every N iterations')
    p.add_argument('--val-stride', type=int, default=Stride_Size, help='sliding-window stride for validation')
    p.add_argument('--iters-per-epoch', type=int, default=None,
                   help='override the 1000 iterations/epoch of ISPRS_dataset (for quick tests)')
    p.add_argument('--amp', action='store_true', help='bf16 autocast for training/inference')
    p.add_argument('--weights', default=None, help='checkpoint to load (Test mode)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tag', default='', help='suffix for log/result names')
    return p.parse_args()


class Tee:
    """Duplicate writes to the console and a log file."""

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, 'a', encoding='utf-8', buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


def build_model(name):
    if name == 'UNetformer':
        return UNetFormer(num_classes=N_CLASSES)
    if name == 'ResNet':
        return ResNetUNet(num_classes=N_CLASSES)
    if name == 'ViT':
        return ViTSeg(num_classes=N_CLASSES, img_size=WINDOW_SIZE[0])
    raise ValueError(name)


SHOW_BARS = sys.stderr.isatty()  # keep tqdm bars out of redirected output


_TEST_CACHE = {}


def load_test_tile(id):
    """Test image (CHW float tensor on GPU) and eroded label, cached across validations."""
    if id not in _TEST_CACHE:
        img = 1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32')
        if DATASET == 'Urban':
            gt_e = np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64') - 1
        else:
            gt_e = convert_from_color(io.imread(ERODED_FOLDER.format(id)))
        _TEST_CACHE[id] = (torch.from_numpy(img.transpose((2, 0, 1)).copy()).cuda(), img.shape[:2], gt_e)
    return _TEST_CACHE[id]


@torch.no_grad()
def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE,
         amp=False, return_metrics=False):
    # Use the network on the test set (sliding window, overlapping logits are summed)
    all_preds = []
    all_gts = []
    for id in tqdm(test_ids, leave=False, disable=not SHOW_BARS):
        img, shape, gt_e = load_test_tile(id)
        pred = torch.zeros((N_CLASSES,) + tuple(shape), device='cuda')

        total = count_sliding_window(np.empty(shape), step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(np.empty(shape), step=stride, window_size=window_size)),
                     total=total, leave=False, disable=not SHOW_BARS)):
            # Build the tensor
            image_patches = torch.stack([img[:, x:x + w, y:y + h] for x, y, w, h in coords])

            # Do the inference
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                outs = net(image_patches)
            outs = outs.float()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                pred[:, x:x + w, y:y + h] += out
            del (outs)

        pred = pred.argmax(dim=0).cpu().numpy()
        all_preds.append(pred)
        all_gts.append(gt_e)

    MIoU, scores = metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel(), return_all=True)
    out = (MIoU, scores) if return_metrics else MIoU
    if all:
        return out, all_preds, all_gts
    return out


def train(net, optimizer, train_loader, epochs, args, run_name, scheduler=None, weights=WEIGHTS):
    weights = weights.cuda()
    best_path = os.path.join(RESULTS_DIR, run_name + '_best.pth')

    iter_ = 0
    MIoU_best, best_epoch, best_iter = -1.0, None, None
    history = []
    t0 = time.time()
    for e in range(1, epochs + 1):

        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
                output = net(data)
            loss = loss_calc(output.float(), target, weights)

            loss.backward()
            optimizer.step()

            if iter_ % 100 == 0:
                pred = np.argmax(output.data.float().cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('[{}] Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLr: {:.6f}\tLoss: {:.6f}\tAccuracy: {:.2f}\tElapsed: {:.1f} min'.format(
                    datetime.datetime.now().strftime('%H:%M:%S'), e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), optimizer.param_groups[0]['lr'], loss.item(),
                    accuracy(pred, gt), (time.time() - t0) / 60))
            iter_ += 1

            del (data, target, loss)

            ## !!! You can increase the frequency of testing to find better models.
            if iter_ % args.val_interval == 0:
                # We validate with the largest possible stride for faster computing
                net.eval()
                tv = time.time()
                MIoU = test(net, test_ids, all=False, stride=args.val_stride, amp=args.amp)
                net.train()
                history.append({'epoch': e, 'iter': iter_, 'mIoU': float(MIoU)})
                msg = 'Validation @ epoch {} iter {}: mIoU {:.4f} ({:.1f} s)'.format(e, iter_, MIoU, time.time() - tv)
                if MIoU > MIoU_best:
                    torch.save(net.state_dict(), best_path)
                    MIoU_best, best_epoch, best_iter = MIoU, e, iter_
                    msg += '  -> new best, saved to {}'.format(os.path.relpath(best_path, ROOT))
                print(msg + '  (best {:.4f})'.format(MIoU_best))

        if scheduler is not None:
            scheduler.step()

    final_path = os.path.join(RESULTS_DIR, run_name + '_final.pth')
    torch.save(net.state_dict(), final_path)
    print('Saved final weights to {}'.format(os.path.relpath(final_path, ROOT)))
    return {'best_mIoU': MIoU_best, 'best_epoch': best_epoch, 'best_iter': best_iter,
            'best_path': best_path, 'final_path': final_path,
            'train_minutes': (time.time() - t0) / 60, 'history': history}


def append_summary(row):
    path = os.path.join(RESULTS_DIR, 'summary.csv')
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = '{}_{}{}'.format(args.model, DATASET, ('_' + args.tag) if args.tag else '')
    log_path = os.path.join(LOG_DIR, '{}_{}_{}.log'.format(run_name, args.mode, stamp))
    sys.stdout = Tee(sys.__stdout__, log_path)

    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print(args.model + ', ' + args.mode + ', ' + DATASET)
        print('Args: ' + json.dumps(vars(args)))
        print('Log file: ' + os.path.relpath(log_path, ROOT))
        print('Device: ' + torch.cuda.get_device_name(0))

        net = build_model(args.model).cuda()
        n_params = sum(p.numel() for p in net.parameters())
        print('Params: {} ({:.2f} M)'.format(n_params, n_params / 1e6))

        if args.mode == 'Train':
            # Load the datasets
            print("training : ", str(len(train_ids)) + ", testing : ", str(len(test_ids)) + ", Stride_Size : ",
                  str(args.val_stride), ", BATCH_SIZE : ", str(BATCH_SIZE))
            train_set = ISPRS_dataset(train_ids, cache=CACHE)
            if args.iters_per_epoch:
                train_set = torch.utils.data.Subset(train_set, range(args.iters_per_epoch * BATCH_SIZE))
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
            # We define the scheduler
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)

            res = train(net, optimizer, train_loader, args.epochs, args, run_name, scheduler)

            # Final evaluation of the last-epoch weights
            print('=== Final evaluation (last epoch weights, stride {}) ==='.format(args.val_stride))
            net.eval()
            final_mIoU, final_scores = test(net, test_ids, stride=args.val_stride, amp=args.amp, return_metrics=True)
            print('=== Best checkpoint: mIoU {:.4f} at epoch {} iter {} ==='.format(
                res['best_mIoU'], res['best_epoch'], res['best_iter']))

            with open(os.path.join(RESULTS_DIR, run_name + '_history.json'), 'w') as f:
                json.dump({'args': vars(args), 'history': res['history'], 'final': final_scores}, f, indent=1)
            append_summary({
                'time': stamp, 'model': args.model, 'dataset': DATASET, 'params_M': round(n_params / 1e6, 2),
                'epochs': args.epochs, 'amp': args.amp,
                'best_mIoU': round(res['best_mIoU'], 4), 'best_epoch': res['best_epoch'], 'best_iter': res['best_iter'],
                'final_mIoU': round(final_scores['mIoU'], 4), 'final_mF1': round(final_scores['mF1'], 4),
                'final_OA': round(final_scores['OA'], 2), 'final_Kappa': round(final_scores['Kappa'], 4),
                'train_minutes': round(res['train_minutes'], 1),
                'best_weights': os.path.relpath(res['best_path'], ROOT),
                'final_weights': os.path.relpath(res['final_path'], ROOT),
                'log': os.path.relpath(log_path, ROOT),
            })
            print('Done. Summary appended to results/summary.csv')

        elif args.mode == 'Test':
            weights_path = args.weights or os.path.join(RESULTS_DIR, run_name + '_best.pth')
            net.load_state_dict(torch.load(weights_path, map_location='cuda'))
            print('Loaded ' + weights_path)
            net.eval()
            (MIoU, scores), all_preds, all_gts = test(net, test_ids, all=True, stride=32, amp=args.amp,
                                                      return_metrics=True)
            print("MIoU: ", MIoU)
            out_dir = os.path.join(RESULTS_DIR, run_name + '_pred')
            os.makedirs(out_dir, exist_ok=True)
            for p, id_ in zip(all_preds, test_ids):
                img = convert_to_color(p)
                io.imsave(os.path.join(out_dir, 'inference_{}_tile_{}.png'.format(args.model, id_)), img,
                          check_contrast=False)
            print('Saved predictions to ' + os.path.relpath(out_dir, ROOT))
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        raise
    finally:
        sys.stdout.flush()


if __name__ == '__main__':
    main()
