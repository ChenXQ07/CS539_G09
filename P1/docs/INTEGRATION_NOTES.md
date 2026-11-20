# Data integration decisions

Reviewed repository commit: `3f27b5c549079ada6135ad6405b62a4ee69cfe71`.

| Topic | Existing final implementation | P1 specification / partner handoff | Action |
| --- | --- | --- | --- |
| Partitions | 12 train, 4 test, no separate validation | P1 fixes 10 train, 2 validation, 4 test | Agree on future experiment protocol before changing training code |
| Checkpoint selection | Repeated evaluation on `test_ids` | Validation tiles select settings/checkpoints | Wire validation separately and identify old results by their original protocol |
| Patch size | 256x256 | P1 proposes 256x256; encoder-reference handoff proposes 512x512 | Confirm one training input contract |
| Scaling and normalization | Divide uint8 values by 255 | Encoder-reference handoff additionally requests ImageNet mean/std | Normalize exactly once using the agreed contract |
| Pretraining | Existing result documentation says pretrained encoders | Earlier proposal describes random initialization | Record actual initialization per run; align the implementation and current report |
| Labels | Black maps to 6, loss ignores 255 | P1 plans black/undefined -> 255, valid classes 0-5 | Implement the explicit mapping before using excluded pixels in loss |
| Metrics | Existing Vaihingen mIoU averages classes 0-4 | Full/eroded masks and support are recorded by P1 | State masking and class averaging consistently |

These are integration decisions, not completed training changes. All compared models should use one stated, reproducible protocol.
