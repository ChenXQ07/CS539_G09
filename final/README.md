# UNetFormer vs. ResNet vs. ViT for Remote Sensing Semantic Segmentation (ISPRS Vaihingen)

## 1. Project structure

| Path | Contents |
|---|---|
| `log/` | Training logs. Everything a run prints (loss, the confusion matrix and metrics from every validation, any errors) is written to `<model>_Vaihingen_Train_<time>.log`. |
| `model/` | Code for the three models: `UNetFormer.py` (UNetFormer), `ResNet.py` (ResNet-50 encoder + U-Net decoder) and `ViT.py` (ViT-B/16 encoder + FPN decoder). |
| `results/summary.csv` | Comparison of the three models, one row per training run: parameter count, best mIoU and its epoch, mIoU / mF1 / OA / Kappa of the final weights, training time, and the weight and log paths. |
| `results/*.pth` | Trained weights. `<model>_Vaihingen_best.pth` has the highest validation mIoU; `<model>_Vaihingen_final.pth` is from the end of epoch 50. |
| `results/*_history.json` | The mIoU from every validation of each model, plus the full metrics of the final weights. |
| `train.py` | Training and testing code. Pick the model with `--model UNetformer\|ResNet\|ViT`; use `--mode Test --weights <path>` to run inference and save prediction maps. |
| `utils.py` | Settings and helpers: data path `FOLDER`, dataset `DATASET`, training/test tile IDs, crop size `WINDOW_SIZE`, `BATCH_SIZE`, validation stride `Stride_Size`, class names and color palette, plus the dataset loader, loss function and evaluation metrics. |
| `run_all.ps1` | Trains the three models one after another. |

Examples:

```powershell
D:\Miniconda\envs\torch_basic\python.exe train.py --model UNetformer
D:\Miniconda\envs\torch_basic\python.exe train.py --model ViT --mode Test --weights results\ViT_Vaihingen_best.pth
```

## 2. Experimental setup

- **Data**: ISPRS Vaihingen. 12 training tiles (1, 3, 23, 26, 7, 11, 13, 28, 17, 32, 34, 37) and 4 test tiles (5, 21, 15, 30).
- **Training**: random 256×256 crops with random flips; batch size 10; 1000 iterations per epoch, 50 epochs.
- **Optimizer**: SGD (lr 0.01, momentum 0.9, weight decay 5e-4); the learning rate is multiplied by 0.1 at epochs 25, 35 and 45. fp32 training, identical for all three models.
- **Validation**: every 500 iterations on the test tiles (100 validations in total), sliding 256×256 windows with stride 32, scored against the eroded (boundary-removed) ground truth.
- **Metrics**: mIoU and mF1 average the first 5 classes (roads, buildings, low vegetation, trees, cars), excluding clutter; OA and Kappa are computed over all pixels.
- **Hardware**: NVIDIA RTX 5080 (16 GB), PyTorch 2.11, timm 1.0.30.

## 3. Results

Tables 1 and 2 report the **best epoch** (the validation with the highest mIoU), taken from that validation's output in `log/`. The `final_*` columns in `results/summary.csv` are the results of the final weights from epoch 50.

> Note: the best epoch is selected on the test tiles and scored on the same tiles, so best-epoch numbers are optimistic. For formal reporting, use the final-weight results in `summary.csv`.

### Table 1: Overall metrics at the best epoch

| Model | Best epoch (iteration) | mIoU | mF1 | OA (%) | Kappa |
|---|---|---|---|---|---|
| UNetFormer | 31 (31000) | **0.8212** | **0.8992** | 91.37 | 0.8838 |
| ResNet | 45 (44500) | 0.8209 | **0.8992** | 91.01 | 0.8790 |
| ViT | 9 (8500) | 0.8055 | 0.8892 | **91.56** | **0.8864** |

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

| Model | Encoder | timm weights | Pretraining data | Decoder (randomly initialized) |
|---|---|---|---|---|
| UNetFormer | ResNet-18 | `resnet18.fb_swsl_ig1b_ft_in1k` | Semi-weakly supervised on 1B Instagram images → fine-tuned on ImageNet-1k | UNetFormer decoder (global-local Transformer blocks + feature refinement head) |
| ResNet | ResNet-50 | `resnet50.a1_in1k` | ImageNet-1k | U-Net decoder (5 upsampling stages + skip connections) |
| ViT | ViT-B/16 | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | ImageNet-21k → fine-tuned on ImageNet-1k (position embeddings interpolated to 256×256) | Simple feature pyramid + FPN (features from blocks 3/6/9/12) |

Only the encoders use pretrained weights; all decoders are trained from random initialization.

## 4. Conclusions

- **UNetFormer**: highest mIoU and mF1, and the smallest parameter count, compute and training time of the three. Best overall.
- **ResNet**: best mIoU only 0.0003 below UNetFormer and the best on cars, but about 3× the parameters and 2× the training time.
- **ViT**: highest OA and Kappa, and the best on low vegetation and trees, but clearly weaker on cars (0.7087). The likely cause is that ViT only extracts features at 1/16 resolution, which hurts small objects; mIoU and mF1 weight every class equally, so they drop. ViT peaked at epoch 9 and its validation scores fell afterwards, probably from overfitting the 12 training tiles; SGD with lr 0.01 may also be too high for a ViT. AdamW with a lower learning rate and warmup is worth trying.
