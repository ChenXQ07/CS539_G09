# P1 Data Preparation and Quality Checks

**Owner:** @JialiangSun1

## Completed work

- Audited 33 original three-channel uint8 IRRG image tiles.
- Verified matching dimensions and valid six-class reference colors for all 16 labeled tiles, including full and eroded references.
- Checked full versus eroded labels outside excluded pixels and compared all 32 local reference files with their matching archive entries.
- Identified the byte-identical duplicate reference for tile 23 and counted it once.
- Prepared reproducible audit scripts, file manifests, source fingerprints and a fixed P1 tile split.
- Documented class support, ignored-pixel requirements and the planned data interface.

This contribution provides data evidence and a split specification. Executable label conversion, the data loader, aligned augmentation and training integration remain next tasks.

## Fixed P1 split for integration

Version: `vaihingen_full16_v1`.

| Partition | Tile IDs | Count |
| --- | --- | --- |
| Training | 1, 3, 7, 11, 13, 17, 23, 26, 28, 32 | 10 |
| Validation | 34, 37 | 2 |
| Test | 5, 15, 21, 30 | 4 |

See [the split specification](splits/vaihingen_full16_v1.json) and [the 16-tile manifest](reports/full16_v1/data_manifest.csv). Split original tiles before cropping. Keep every crop, augmentation and auxiliary view in its tile partition. Use training/validation only for parameter updates, tuning and checkpoint selection as appropriate. Reserve test tiles for final evaluation.

### Integration status

The reviewed repository commit `3f27b5c549079ada6135ad6405b62a4ee69cfe71` uses 12 training tiles and 4 test tiles in `final/utils.py`; `final/train.py` also uses those test tiles for checkpoint selection. The existing published runs therefore have a different protocol. Adding this P1 split does not convert those results into independent 10/2/4 test results. Keep previous results identified by their original protocol, and use an agreed split consistently for future comparisons.

Preprocessing must also be reconciled before implementation: `final/` currently uses 256x256 patches scaled to [0,1], while the P4 encoder-reference handoff proposes 512x512 patches with ImageNet normalization. [Integration notes](docs/INTEGRATION_NOTES.md) list the decisions still needed.

## Files

| Path | Purpose |
| --- | --- |
| `audit_data.py` | Shared palette, metadata and file-check helpers, plus a single-folder audit command |
| `inspect_data_bundle.py` | Audit imagery/labels across split export folders and inspect auxiliary products |
| `freeze_full_split.py` | Verify source hashes and generate or check the complete-data split artifacts |
| `artifact_io.py` | Prevent accidental replacement of an existing fixed data specification |
| `splits/` | Shared tile IDs, data rules, source hashes and class support |
| `reports/data_bundle/` | File inventory and reference-consistency evidence |
| `reports/full16_v1/` | Current 16-tile manifest and planned data contract |
| `tests/` | Palette, pairing, tile-separation and immutable-artifact checks |

## Run the checks

From the repository root:

```bash
python -m pip install -r P1/requirements-data.txt
python -m unittest discover -s P1/tests -v
```

The audit evidence records the versions used. These scripts require NumPy and Pillow, without importing the model training environment.

## Reproduce with local data

Keep the dataset outside Git. The data-root argument points to the directory containing the supplied data fragments, or an equivalent directory containing the usual `top`, `gts_for_participants`, `gts_eroded_for_participants` and auxiliary folders.

```bash
python -m P1.inspect_data_bundle --data-root "PATH/TO/Data" --output-dir P1/local_outputs/data_bundle
python -m P1.freeze_full_split --data-root "PATH/TO/Data" --audit-path P1/local_outputs/data_bundle/bundle_summary.json --split-path P1/local_outputs/vaihingen_full16_v1.json --output-dir P1/local_outputs/full16_v1
```

Published manifests record the audited file layout using paths relative to the data root. If your layout differs, use the locally generated manifest above. The tile assignment stays fixed; existing split artifacts are never silently overwritten.

## Planned data contract

- Input bands: NIR, red and green, in that order.
- Image arrays: three-channel uint8, planned conversion to float32 in [0,1].
- Training labels: full reference masks. Eroded references remain separately available for the stated evaluation protocol.
- IDs: 0 impervious surfaces, 1 buildings, 2 low vegetation, 3 trees, 4 cars, 5 clutter/background.
- Planned ignore index: 255 for black/undefined pixels. Unknown colors must be investigated rather than silently mapped to class 0.
- Planned training transforms: aligned 256x256 image/label crops and synchronized horizontal/vertical flips.
- Validation and test processing: deterministic and independent of training augmentation.

The validation references have no class-5 ground-truth support. Report per-class support and the exact masking/averaging convention when computing metrics. SAM/noise/prediction products are not accepted as ground-truth references by this contribution. Optional DSM paths are recorded but no height/SAM input is enabled.

Audit checks establish local file consistency. They do not independently certify annotation provenance or pixel-perfect annotation quality.

## Next tasks

By November 25: complete and test label conversion, preprocessing and data loading, reconcile the interface with the training owners, and provide reproducible integration examples.
