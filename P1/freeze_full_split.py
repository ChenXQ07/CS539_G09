"""Freeze the complete-data tile split and write current P1 handoff evidence.

python -m P1.freeze_full_split --data-root PATH/TO/Data

The complete-data version is separate from any earlier debugging subsets.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
from pathlib import Path

from P1.audit_data import EXPECTED_LABELED_IDS, sha256_file
from P1.artifact_io import write_immutable

SPLIT_ID = "vaihingen_full16_v1"
DECISION_DATE = "2026-11-20"
PARTITIONS = {
    "train": [1, 3, 7, 11, 13, 17, 23, 26, 28, 32],
    "validation": [34, 37],
    "test": [5, 15, 21, 30],
}


def validate_full_split(partitions: dict, records: dict) -> None:
    if set(partitions) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test partitions")
    if [len(partitions[k]) for k in ("train", "validation", "test")] != [10, 2, 4]:
        raise ValueError("Expected the agreed 10/2/4 tile counts")
    ids = [n for group in partitions.values() for n in group]
    if len(ids) != len(set(ids)):
        raise ValueError("Tile leakage: an ID occurs in multiple partitions")
    if set(ids) != set(EXPECTED_LABELED_IDS):
        raise ValueError("The split must cover all 16 labeled tile IDs exactly once")
    if any(n not in records or records[n].get("pair_status") != "pass" for n in ids):
        raise ValueError("A selected tile lacks passing image/reference checks")


def canonical_full_label(record: dict) -> str:
    return min(record["full_label_paths"], key=lambda p: ("(" in Path(p).name, p))


def build_handoff(data_root: Path, audit_path: Path, split_path: Path, output: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    records = {r["tile_id"]: r for r in audit["labeled_tiles"]}
    images = {r["tile_id"]: r for r in audit["original_images"]}
    heights = {r["tile_id"]: r for r in audit["dsm_checks"]}
    validate_full_split(PARTITIONS, records)
    rows = []
    for split, ids in PARTITIONS.items():
        for n in ids:
            r, image = records[n], images[n]
            full_path, eroded_path = canonical_full_label(r), r["eroded_label_paths"][0]
            for relative, expected in [(image["path"], image["sha256"]),
                                       (full_path, r["full_label_sha256"]),
                                       (eroded_path, r["eroded_label_sha256"])]:
                if sha256_file(data_root / relative) != expected:
                    raise ValueError(f"Source changed since inspection: {relative}")
            rows.append({"tile_id": n, "split": split, "image_path": image["path"],
                         "full_label_path": full_path, "eroded_label_path": eroded_path,
                         "width": r["image_size"][0], "height": r["image_size"][1],
                         "image_sha256": image["sha256"], "full_label_sha256": r["full_label_sha256"],
                         "eroded_label_sha256": r["eroded_label_sha256"],
                         "class_pixels_full": r["full_label_stats"]["class_pixels"],
                         "class_pixels_eroded": r["eroded_label_stats"]["class_pixels"],
                         "ignored_pixels_full": r["full_label_stats"]["ignored_pixels"],
                         "ignored_pixels_eroded": r["eroded_label_stats"]["ignored_pixels"],
                         "dsm_path_optional": heights.get(n, {}).get("path", "")})
    support = {}
    for split, ids in PARTITIONS.items():
        support[split] = {}
        for reference in ("full", "eroded"):
            counts = {str(c): sum(records[n][f"{reference}_label_stats"]["class_pixels"][str(c)] for n in ids) for c in range(6)}
            support[split][reference] = {"class_pixels": counts,
                                        "classes_with_gt_pixels": [int(c) for c, count in counts.items() if count > 0]}
    contract = {
        "input_channels": ["near_infrared", "red", "green"],
        "image_scaling": "uint8 / 255 -> float32 in [0, 1]",
        "training_reference": "full_label_path", "eroded_reference": "eroded_label_path",
        "class_ids": {"0": "impervious_surfaces", "1": "buildings", "2": "low_vegetation", "3": "trees", "4": "cars", "5": "clutter_background"},
        "planned_ignore_index": 255,
        "training_crop_size": [256, 256],
        "planned_augmentation": "Synchronized horizontal and vertical flips on training images and labels only",
        "validation_test": "Deterministic image/label handling; never update model parameters from these tiles",
        "optional_height_input": "DSM paths are recorded for future workflow decisions; DSM/SAM products are not enabled as segmentation inputs here",
    }
    split = {
        "split_id": SPLIT_ID, "decision_date": DECISION_DATE,
        "status": "fixed_complete_labeled_dataset", "partition_unit": "original_image_tile",
        "partitions": PARTITIONS, "counts": {k: len(v) for k, v in PARTITIONS.items()},
        "selection_rationale": "Keep test IDs 5, 15, 21 and 30 from the original project configuration. Reserve 34 and 37 from the remaining 12 labeled tiles for validation. Use the other 10 for training. This is a fixed assignment, not selected using model scores.",
        "source_checks": "All 16 triples passed dimension, palette and full-versus-eroded consistency checks. Source hashes were rechecked when this split was generated.",
        "source_audit_sha256": sha256_file(audit_path),
        "class_support": support, "data_contract": contract,
        "rules": ["Keep every crop, augmentation and derived view in its original tile's partition.",
                  "Use the same split for all compared models and all connected neural stages.",
                  "Select settings and checkpoints using training/validation data only; reserve test tiles for final evaluation.",
                  "Keep full and eroded references distinct and state the evaluation mask and class averaging convention.",
                  "Save a new version if a later experiment changes the tile assignment."],
        "label_metadata_note": "Valid colors and archive byte agreement verify file consistency, not independent source provenance or pixel-perfect annotation accuracy.",
        "tile_files": [{k: v for k, v in row.items() if not k.startswith("class_pixels") and not k.startswith("ignored_pixels")} for row in rows],
    }
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()} for row in rows)

    contribution = (
        "Contributions to date\n\n"
        "P1: Audited the Vaihingen data and verified 33 original IRRG image tiles and 16 matching sets of full and eroded reference labels. "
        "Checked image-label dimensions, label colors, duplicate files, and consistency between full and eroded references. "
        "Prepared reproducible audit scripts, file manifests, and a fixed tile-level split of 10 training, 2 validation, and 4 test tiles. "
        "Documented class support, ignored-pixel handling requirements, and a shared data specification for integration.\n\n"
        "Next tasks\n\n"
        "Implement six-class color-to-ID conversion and ignore-index handling with focused checks — P1 — November 21, 2026.\n"
        "Implement split-aware data loading with aligned 256×256 crops and training-only flips — P1 — November 22, 2026.\n"
        "Verify deterministic validation/test handling and produce preprocessing examples — P1 — November 24, 2026.\n"
        "Integrate the data interface with the team's training/evaluation code — P1, with support from the training and evaluation owners — November 25, 2026.\n"
        "Finalize data documentation and a reproducible handoff for system integration — P1 — November 29, 2026.\n"
    )
    report = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>P1 data handoff</title>",
              "<style>body{font:17px system-ui;max-width:1150px;margin:36px auto;padding:0 24px;color:#193047}p{line-height:1.6}table{border-collapse:collapse;width:100%;margin:22px 0}th,td{border:1px solid #ccd5de;padding:12px;text-align:left}th{background:#eef3f7}pre{white-space:pre-wrap;font:inherit;line-height:1.6}</style>",
              "<h1>P1 complete-data audit and fixed split</h1>",
              "<p>Completed: data inventory, image/reference checks, duplicate handling, source fingerprints, and a fixed 10/2/4 tile partition. Remaining: executable label conversion, data loading, aligned augmentation, and integration.</p>",
              "<table><tr><th>Partition</th><th>Tile IDs</th><th>Count</th></tr>"]
    for name, ids in PARTITIONS.items():
        report.append(f"<tr><td>{name}</td><td>{', '.join(map(str,ids))}</td><td>{len(ids)}</td></tr>")
    report.append("</table><h2>Evidence</h2><p>33 original images decode as three-channel uint8. All 16 labeled triples have matching dimensions and valid palettes. Full and eroded references agree outside excluded pixels. All 32 local reference files match their corresponding members in v.zip. The duplicate area23 reference has identical bytes and is counted once.</p>")
    report.append("<h2>Class support</h2><p>Class 5 is clutter/background. State reference masking and averaging explicitly when reporting metrics. Validation contains no class-5 reference pixels.</p><table><tr><th>Partition</th><th>Full-reference classes with support</th><th>Eroded-reference classes with support</th></tr>")
    for name, values in support.items():
        report.append(f"<tr><td>{name}</td><td>{values['full']['classes_with_gt_pixels']}</td><td>{values['eroded']['classes_with_gt_pixels']}</td></tr>")
    report.extend(["</table><h2>Report text and remaining tasks</h2>", f"<pre>{html.escape(contribution)}</pre>",
                   "<p>Only P1's contribution is described here. Other team members should supply their own completed work.</p>",
                   "<p><a href='data_manifest.csv'>Current 16-tile manifest</a> · <a href='progress_sections.txt'>Copyable progress-report text</a> · <a href='data_contract.json'>Shared data specification</a></p></html>"])
    artifacts = {split_path: json.dumps(split, ensure_ascii=False, indent=2) + "\n",
                 output / "data_manifest.csv": buffer.getvalue(),
                 output / "data_contract.json": json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
                 output / "progress_sections.txt": contribution,
                 output / "handoff_report.html": "\n".join(report)}
    for path, content in artifacts.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Create a new version rather than changing {path}")
    for path, content in artifacts.items():
        write_immutable(path, content)
    print(json.dumps({"split_id": SPLIT_ID, "partitions": PARTITIONS,
                      "verified_source_files": 48, "class_support": {k: {r: v[r]['classes_with_gt_pixels'] for r in v} for k,v in support.items()}}, ensure_ascii=False))
    return split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--audit-path", type=Path, default=Path("P1/reports/data_bundle/bundle_summary.json"))
    parser.add_argument("--split-path", type=Path, default=Path("P1/splits/vaihingen_full16_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("P1/reports/full16_v1"))
    args = parser.parse_args()
    build_handoff(args.data_root, args.audit_path, args.split_path, args.output_dir)
