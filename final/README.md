## 1. Project structure

| Path | Contents |
|---|---|
| `log/` | Training logs. |
| `model/` | Code for the three models: `UNetFormer.py`, `ResNet.py` and `ViT.py`. |
| `summary.csv` | Comparison of the three models (raw file)|
| `train.py` | Training and testing code.|
| `utils.py` | Settings. |

## 2. Experimental setup

- **Data**: ISPRS Vaihingen. 12 training tiles (1, 3, 23, 26, 7, 11, 13, 28, 17, 32, 34, 37) and 4 test tiles (5, 21, 15, 30).
- **Training**: random 256×256 crops with random flips; batch size 10; 1000 iterations per epoch, 50 epochs.
- **Optimizer**: SGD (lr 0.01, momentum 0.9, weight decay 5e-4); the learning rate is multiplied by 0.1 at epochs 25, 35 and 45. fp32 training, identical for all three models.
- **Validation**: every 500 iterations on the test tiles (100 validations in total), sliding 256×256 windows with stride 32, scored against the eroded (boundary-removed) ground truth.
- **Metrics**: mIoU and mF1 average the first 5 classes (roads, buildings, low vegetation, trees, cars), excluding clutter; OA and Kappa are computed over all pixels.
- **Hardware**: NVIDIA RTX 5080 (16 GB), PyTorch 2.11, timm 1.0.30.

## 3. Results

Tables 1 and 2 report the **best epoch** (the validation with the highest mIoU).

### Table 1: Overall metrics at the best epoch

| Model | Best epoch | mIoU | mF1 | OA (%) | Kappa |
|---|---|---|---|---|---|
| UNetFormer | 31 | **0.8212** | **0.8992** | 91.37 | 0.8838 |
| ResNet | 45 | 0.8209 | **0.8992** | 91.01 | 0.8790 |
| ViT | 9 | 0.8055 | 0.8892 | **91.56** | **0.8864** |

### Table 2: Per-class IoU at the best epoch

| Model | Roads | Buildings | Low veg. | Trees | Cars | mIoU |
|---|---|---|---|---|---|---|
| UNetFormer | **0.8633** | **0.9378** | 0.6676 | 0.8320 | 0.8055 | **0.8212** |
| ResNet | 0.8522 | 0.9261 | 0.6694 | 0.8302 | **0.8265** | 0.8209 |
| ViT | 0.8611 | 0.9334 | **0.6841** | **0.8404** | 0.7087 | 0.8055 |

### Table 3: Model complexity

| Model | Params (M) | FLOPs (G) | Inference (img/s) | Peak training memory (GB) | Total training time (min) | Time per validation (s) | Weight file (MB) |
|---|---|---|---|---|---|---|---|
| UNetFormer | **11.69** | **2.93** | **1650** | **0.64** | **41.0** | **11.9** | **47.0** |
| ResNet | 32.52 | 10.63 | 731 | 1.93 | 83.0 | 25.8 | 130.4 |
| ViT | 93.77 | 37.35 | 302 | 3.50 | 182.8 | 62.0 | 375.2 |

- **FLOPs**: one 256×256 input, counted as multiply-accumulates (MACs) with `torch.utils.flop_counter`.
- **Inference**: batch 10, 256×256, fp32 forward pass, measured on the RTX 5080.
- **Peak training memory**: peak of one forward + backward pass at batch 10, excluding optimizer state.
- **Total training time**: 50 epochs, including the 100 validations.
- **Time per validation**: 4 test tiles at stride 32; median over the 100 validations.

### Table 4: Pretrained encoders and pretraining data

| Model | Encoder | timm weights | Pretraining data |
|---|---|---|---|
| UNetFormer | ResNet-18 | `resnet18.fb_swsl_ig1b_ft_in1k` | Semi-weakly supervised on 1B Instagram images → fine-tuned on ImageNet-1k |
| ResNet | ResNet-50 | `resnet50.a1_in1k` | ImageNet-1k |
| ViT | ViT-B/16 | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | ImageNet-21k → fine-tuned on ImageNet-1k (position embeddings interpolated to 256×256) |

Only the encoders use pretrained weights; all decoders are trained from random initialization.

## 4. Conclusions

- **UNetFormer**: highest mIoU and mF1, and the smallest parameter count, compute and training time of the three. Best overall.
- **ResNet**: best mIoU only 0.0003 below UNetFormer and the best on cars, but about 3× the parameters and 2× the training time.
- **ViT**: highest OA and Kappa, and the best on low vegetation and trees, but clearly weaker on cars (0.7087). The likely cause is that ViT only extracts features at 1/16 resolution, which hurts small objects; mIoU and mF1 weight every class equally, so they drop. ViT peaked at epoch 9 and its validation scores fell afterwards, probably from overfitting the 12 training tiles; SGD with lr 0.01 may also be too high for a ViT. AdamW with a lower learning rate and warmup is worth trying.
