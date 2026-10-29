# Urban Semantic Segmentation with UNetFormer

CS539 Group 09 project using the ISPRS Vaihingen dataset and the supplied RSSS_code implementation.

## Project overview

We aim to turn three-channel aerial imagery into a pixel-level land-cover map. The input is Vaihingen IRRG imagery (near infrared, red, and green); the output is a map of the same spatial extent with six classes: impervious surfaces, buildings, low vegetation, trees, cars, and clutter/background.

We will adapt and train a ResNet18-based UNetFormer model for a selected Vaihingen subset. The planned deliverables are a reproducible data and training workflow, trained model checkpoints, full-tile predictions, evaluation results, and presentation materials.

**Status:** project planning and repository organization. The architecture and training settings below describe the proposed work; they are not completed experiments or reported results. Dataset partitions, the baseline, and the controlled comparison remain to be finalized.

## Research basis and design choice

CNN-based approaches such as U-Net extract hierarchical spatial features [1]. Transformer-based approaches such as Segmenter use attention to model contextual relationships [2]. Standard global attention can be costly for high-resolution imagery. We select UNetFormer, which combines a lightweight ResNet18 encoder with a decoder using local convolutions, window-based attention, and cross-window context aggregation [3]. This design balances spatial detail, contextual modelling, and computational cost.

We will reuse the encoder, global-local attention decoder, feature fusion, and sliding-window inference in the supplied implementation. Our adaptations focus on dataset configuration, label handling, and training for the selected data subset. The planned inputs exclude SAM and height data. Main limitations are the context available within each patch and the domain difference between natural-image pretraining and IRRG imagery.

U-Net and Segmenter motivate the design choice; they are not yet designated as the experimental baseline.

## Architecture and interfaces

The system has two connected neural components followed by a deterministic reconstruction component. Data preparation supplies image patches and the metadata needed to reconstruct the original tile.

~~~mermaid
flowchart LR
    D["Data preparation<br/>IRRG patches: B x 3 x 256 x 256"] --> E["ResNet18 encoder<br/>Pretrained and fine-tuned"]
    E -->|Multiscale features| F["UNetFormer decoder<br/>Random initialization and team training"]
    F -->|"Logits: B x 6 x 256 x 256"| R["Window fusion and reconstruction<br/>Deterministic; no trainable parameters"]
    D -. Patch coordinates and original tile size .-> R
    R --> O["Class map: H x W<br/>Six classes"]
~~~

| Stage | Input | Output and next consumer | Training status |
| --- | --- | --- | --- |
| Data preparation | Paired IRRG tiles and reference labels | Aligned image/label patches for training; image patches for the encoder; coordinates and original dimensions for reconstruction | Data processing, no trainable parameters |
| ResNet18 encoder | B x 3 x 256 x 256 image tensor | Feature maps of 64 x 64 x 64, 128 x 32 x 32, 256 x 16 x 16, and 512 x 8 x 8, passed to the decoder | Pretrained initialization; fine-tuned by the team |
| UNetFormer decoder | Multiscale encoder features | B x 6 x 256 x 256 class logits, passed to reconstruction during inference | Random initialization; trained by the team |
| Window fusion and reconstruction | Patch logits, coordinates, and original tile dimensions | H x W class map for evaluation and visualization | Deterministic; no trainable parameters |

B is batch size. Encoder feature shapes use C x H x W and omit the batch dimension. The decoder includes three global-local attention blocks, weighted feature fusion and skip connections, feature refinement, and an output head. Reconstruction places predictions at their recorded coordinates, sums overlapping class logits, and applies pixelwise argmax.

The encoder and decoder are parts of one jointly trained network. Architecture stages are distinct from the team workspaces below. Course-topic labels will be confirmed against the course syllabus.

## Data preparation

- Use paired Vaihingen IRRG imagery and verified pixel-level reference labels from the official dataset [4].
- Separate training, development, and final-evaluation tiles before extracting patches. Exact tile IDs and the selected subset will be documented before experiments.
- Extract aligned 256 x 256 image and label crops, scale image values to [0, 1], and apply synchronized random horizontal and vertical flips during training.
- Map reference colors to six class indices. Map invalid pixels, including excluded boundaries when eroded references are used, to an ignore label.
- Check image-label alignment, label provenance, valid class indices, and partition separation. Retain patch coordinates and original image dimensions for reconstruction.

## Proposed training plan

We will jointly fine-tune the pretrained ResNet18 encoder and train the decoder from random initialization. Neither neural component will be frozen. Pixelwise cross-entropy will backpropagate through both components; ignored pixels will contribute no loss. Reconstruction remains deterministic.

| Setting | Initial plan |
| --- | --- |
| Loss | Pixelwise cross-entropy with ignored labels excluded |
| Optimizer | SGD |
| Initial learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Batch size | 4 |
| Training budget | Up to 20 epochs, 250 minibatches per epoch |
| Learning-rate schedule | Multiply by 0.1 after epochs 10 and 15 |
| Development evaluation | After each epoch |
| Checkpoint selection | Highest development mIoU |

A short pilot will check GPU memory and runtime before the budget is confirmed. It will also verify gradient updates in both neural components, ignored-label handling, and checkpoint selection. The team will study window-based attention, cross-window context aggregation, multiscale feature fusion, and transfer learning for IRRG imagery.

## Evaluation plan and open decisions

Development mIoU will guide checkpoint selection. Final-evaluation tiles will be reserved until the model and settings are fixed. The baseline model, the single factor in the controlled comparison, exact tile partitions, and the class/ignore policy for reported metrics must be specified before comparison runs. No accuracy or runtime results are claimed at this stage.

## Repository organization and responsibilities

| Workspace | Main responsibility | Deliverables |
| --- | --- | --- |
| [P1/](P1/) | Data preparation | Data access instructions, preprocessing, and partition records |
| [P2/](P2/) | Model implementation and training | Model code, training configurations, and run summaries |
| [P3/](P3/) | Experimental evaluation | Baseline and controlled comparison, metrics, and visual results |
| [P4/](P4/) | Project integration and reporting | Integration notes, report contributions, and presentation materials |
| [final/](final/) | Shared integrated deliverable | Reviewed code, evaluation outputs, and presentation materials |

Members will be assigned to P1-P4 by the team. Keep the Canvas-name/GitHub-account identity mapping in the private Canvas PDF.

## Timeline

All dates are in 2026. Intermediate targets are proposed team checkpoints.

| Date | Milestone | Responsibility |
| --- | --- | --- |
| Nov 1 | Submit proposal and confirm repository organization and owners | All |
| Nov 8 | Prepare data and complete an initial connected-system pilot | P1-P4 |
| Nov 22 | Progress checkpoint with initial results and remaining risks | All |
| Nov 29 | Complete planned training, comparison, and evaluation | All |
| Dec 6 | Review-ready repository and presentation slides | All |
| Dec 9 | Final project submission after team review | All |

## Reproducibility and collaboration

Use separate branches for changes and pull requests for team review. Integrate reviewed contributions in final/. As implementation progresses, document software versions, data partitions, random seeds, configurations, checkpoint locations, and commands to reproduce training and evaluation.

Environment installation and execution instructions will be added after the supplied code has been adapted and the pilot has passed. Download data from the official provider and keep datasets, large weights, credentials, and the private Canvas proposal outside this public repository.

## References

[1] Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 234-241. https://doi.org/10.1007/978-3-319-24574-4_28

[2] Strudel, R., Garcia, R., Laptev, I., and Schmid, C. (2021). Segmenter: Transformer for Semantic Segmentation. ICCV, 7262-7272. https://openaccess.thecvf.com/content/ICCV2021/html/Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.html

[3] Wang, L., Li, R., Zhang, C., Fang, S., Duan, C., Meng, X., and Atkinson, P. M. (2022). UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 190, 196-214. https://doi.org/10.1016/j.isprsjprs.2022.06.008

[4] ISPRS. 2D Semantic Labeling - Vaihingen Data. https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx
