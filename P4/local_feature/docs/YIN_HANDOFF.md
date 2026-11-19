# Interface handoff to P2 and P3

P4 input/output: floating [B,64,h,w], same device. Pass the SAME prepared block-normalized feature to both branches. Each of the three blocks has independent local parameters. P4 returns the unpadded sum of 1x1 Conv-BN and 3x3 Conv-BN; no ReLU, common projection or residual is applied.

P3 must return its spatially aggregated global feature cropped to original h/w, immediately BEFORE the supplied source's `out = out + local`. Then shared code adds the branches and performs pad_out -> common proj -> crop exactly once. The shared Block owns normalization, residuals and MLP. Do not duplicate these inside either branch.

```python
local = local_branch(x)
global_context = global_branch(x)  # P3 API to be agreed
combined = global_context + local
# Apply original common pad/projection/crop and block operations afterward.
```

Request from P3: module and API, global output before common fusion/projection, supported sizes, and seeded example tensor. Request from P2: actual decoder feature preparation, projection/normalization ownership, source version and training entry point. Request from P1: verified batch/masks, IRRG band and scaling policy, class/ignore IDs, tile splits and patch positions.

The submitted proposal uses random initialization, 256x256 crops and scaled [0,1] imagery. Use encoder weights=None and confirm any normalization or size change with the team. Do not detach encoder features during joint training. Branch tests use a synthetic objective; real segmentation loss belongs to the full model.
