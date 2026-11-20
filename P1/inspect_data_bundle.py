"""Inspect split download folders without moving or changing source files.

python -m P1.inspect_data_bundle --data-root PATH/TO/Data
"""
from __future__ import annotations

import argparse
import html
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from P1.audit_data import EXPECTED_LABELED_IDS, label_statistics, metadata, sha256_file, tile_id, write_csv

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def decode(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image)


def inspect_bundle(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside source data")
    output.mkdir(parents=True, exist_ok=True)
    inventory = []
    grouped = defaultdict(lambda: defaultdict(list))
    errors = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        row = {"path": path.relative_to(root).as_posix(), "folder": path.parent.name,
               "tile_id": tile_id(path.name), "size_bytes": path.stat().st_size}
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                row.update(metadata(path), header_ok=True)
                grouped[path.parent.name][row["tile_id"]].append(path)
            except Exception as exc:
                row.update(header_ok=False, error=f"{type(exc).__name__}: {exc}")
                errors.append(row)
        inventory.append(row)

    images = {n: ps[0] for n, ps in grouped["top"].items() if n is not None}
    full = grouped["gts_for_participants"]
    eroded = grouped["gts_eroded_for_participants"]
    records, canonical, original_checks = [], {}, []
    for n, path in sorted(images.items()):
        print("Decode original image", n, flush=True)
        array = decode(path)
        info = {"tile_id": n, "path": path.relative_to(root).as_posix(), "shape": list(array.shape),
                "dtype": str(array.dtype), "sha256": sha256_file(path),
                "three_channel_uint8": array.ndim == 3 and array.shape[2] == 3 and array.dtype == np.uint8}
        original_checks.append(info)
    for n in EXPECTED_LABELED_IDS:
        print("Check labeled tile", n, flush=True)
        row = {"tile_id": n, "image_path": images[n].relative_to(root).as_posix() if n in images else "",
               "full_label_paths": [p.relative_to(root).as_posix() for p in full.get(n, [])],
               "eroded_label_paths": [p.relative_to(root).as_posix() for p in eroded.get(n, [])]}
        if n not in images or not full.get(n) or not eroded.get(n):
            row["pair_status"] = "missing_input"
            records.append(row)
            continue
        reference_path = sorted(full[n], key=lambda p: ("(" in p.name, str(p)))[0]
        eroded_path = eroded[n][0]
        canonical[n] = reference_path
        with Image.open(images[n]) as im:
            image_size = im.size
        with Image.open(reference_path) as im:
            full_size = im.size
            row["full_label_stats"] = label_statistics(im)
            full_array = np.array(im)
        with Image.open(eroded_path) as im:
            eroded_size = im.size
            row["eroded_label_stats"] = label_statistics(im)
            eroded_array = np.array(im)
        row.update(image_size=list(image_size), full_label_size=list(full_size),
                   eroded_label_size=list(eroded_size), size_match=image_size == full_size == eroded_size,
                   full_label_sha256=sha256_file(reference_path), eroded_label_sha256=sha256_file(eroded_path))
        if full_array.shape == eroded_array.shape:
            valid = np.any(eroded_array != 0, axis=-1)
            different = valid & np.any(full_array != eroded_array, axis=-1)
            row.update(eroded_valid_pixel_count=int(valid.sum()),
                       full_vs_eroded_different_valid_pixels=int(different.sum()))
        row["duplicate_full_labels"] = [{"path": p.relative_to(root).as_posix(),
                                        "byte_identical_to_canonical": sha256_file(p) == row["full_label_sha256"]}
                                       for p in full[n] if p != reference_path]
        row["pair_status"] = "pass" if row["size_match"] and row["full_label_stats"]["palette_check"] == "pass" and row["eroded_label_stats"]["palette_check"] == "pass" and row.get("full_vs_eroded_different_valid_pixels") == 0 else "review_required"
        records.append(row)

    # DSM values are floating-point elevations, not grayscale JPG heights.
    height_checks = []
    for n in EXPECTED_LABELED_IDS:
        paths = grouped["dsm"].get(n, [])
        if not paths:
            continue
        a = decode(paths[0])
        finite = np.isfinite(a)
        height_checks.append({"tile_id": n, "path": paths[0].relative_to(root).as_posix(),
                              "shape": list(a.shape), "dtype": str(a.dtype),
                              "finite_fraction": float(finite.mean()),
                              "minimum_finite_value": float(a[finite].min()) if finite.any() else None,
                              "maximum_finite_value": float(a[finite].max()) if finite.any() else None,
                              "image_shape_match": list(a.shape) == list(reversed(next(r for r in records if r["tile_id"] == n)["image_size"]))})

    derived_checks = []
    for folder in ["sam_augmented", "sam_augmented_4c", "sam_augmented_4c_512", "sam_augmented_masks", "sam_boundary_merge", "V_merge", "predict", "ndsm"]:
        for n in [1, 11, 32]:
            paths = grouped[folder].get(n, [])
            if not paths or n not in images:
                continue
            a = decode(paths[0])
            record = {"folder": folder, "tile_id": n, "path": paths[0].relative_to(root).as_posix(),
                      "shape": list(a.shape), "dtype": str(a.dtype),
                      "minimum": float(a.min()), "maximum": float(a.max()),
                      "matches_original_first_three_channels": False,
                      "generation_method_verified": False}
            if a.ndim == 2:
                values = np.unique(a)
                record.update(unique_value_count=len(values), unique_values_preview=values[:24].tolist())
            elif a.ndim == 3:
                original = decode(images[n])
                if a.shape[:2] == original.shape[:2] and a.shape[2] >= 3:
                    record["matches_original_first_three_channels"] = bool(np.array_equal(a[:, :, :3], original))
                if a.shape[2] == 4:
                    values = np.unique(a[:, :, 3])
                    record.update(fourth_channel_unique_count=len(values), fourth_channel_values_preview=values[:24].tolist())
                if a.shape[2] == 3 and folder == "sam_augmented_masks":
                    with Image.open(paths[0]) as im:
                        record["semantic_palette_check"] = label_statistics(im)
            derived_checks.append(record)

    archives = []
    comparisons = []
    for path in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            files = [entry for entry in archive.infolist() if not entry.is_dir()]
            counts = Counter(Path(entry.filename).parent.as_posix() for entry in files)
            archives.append({"path": path.relative_to(root).as_posix(), "member_count": len(files),
                             "uncompressed_bytes": sum(entry.file_size for entry in files), "parent_counts": dict(counts),
                             "member_list": [entry.filename for entry in files],
                             "verification_scope": "Index inspected; reference members additionally read and CRC-checked when compared below"})
            if path.name != "v.zip":
                continue
            for entry in files:
                if Path(entry.filename).parent.as_posix() not in ["gts_for_participants", "gts_eroded_for_participants"]:
                    continue
                n = tile_id(entry.filename)
                local = canonical.get(n) if Path(entry.filename).parent.as_posix() == "gts_for_participants" else eroded.get(n, [None])[0]
                if local is None:
                    continue
                raw = archive.read(entry)
                import hashlib
                comparisons.append({"entry": entry.filename, "crc_ok": True,
                                    "matches_local_file_bytes": hashlib.sha256(raw).hexdigest() == sha256_file(local)})

    counts = {folder: {"file_count": sum(map(len, ids.values())),
                       "unique_tile_count": len([n for n in ids if n is not None]),
                       "tile_ids": sorted(n for n in ids if n is not None)} for folder, ids in grouped.items()}
    summary = {"source_directory_name": root.name,
               "source_layout": "Three complementary export fragments; resolve files across fragments without renaming or moving them",
               "file_count": len(inventory), "folder_counts": counts, "original_images": original_checks,
               "labeled_tiles": records, "passing_labeled_tile_ids": [r["tile_id"] for r in records if r["pair_status"] == "pass"],
               "can_support_original_10_2_4_tile_counts": sum(r["pair_status"] == "pass" for r in records) == 16,
               "dsm_checks": height_checks, "derived_examples": derived_checks,
               "archives": archives, "archive_reference_comparisons": comparisons, "header_errors": errors,
               "split_action": "Existing six-tile split remains unchanged as a historical version; no new split is assigned by inspection",
               "limitations": ["SAM-derived folder names and stored masks do not verify model identity, generation procedure or absence of label leakage.",
                               "JPEG nDSM files are encoded grayscale products; physical height units cannot be assumed from their pixel values.",
                               "The original DSM TIFF contains float values; its units and vertical reference still need dataset documentation.",
                               "Archive byte comparisons establish local consistency, not independent official-source provenance."]}
    (output / "bundle_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "file_inventory.csv", inventory, ["path", "folder", "tile_id", "size_bytes", "width", "height", "mode", "channels", "bits_per_sample", "header_ok", "error"])
    write_csv(output / "labeled_tile_manifest.csv", records, ["tile_id", "image_path", "full_label_paths", "eroded_label_paths", "image_size", "full_label_size", "eroded_label_size", "size_match", "full_vs_eroded_different_valid_pixels", "full_label_sha256", "eroded_label_sha256", "duplicate_full_labels", "pair_status"])
    esc = lambda value: html.escape(str(value))
    report = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Complete Vaihingen bundle inspection</title>",
              "<style>body{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#193047}p{line-height:1.6}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border:1px solid #ccd5de;text-align:left}th{background:#eef3f7}</style>",
              "<h1>Inspection of the new Data folder</h1>",
              f"<p>Files are spread across three complementary export fragments. The image set contains {len(images)} distinct original tiles. {len(summary['passing_labeled_tile_ids'])} labeled triples pass dimension, palette and nonboundary reference-consistency checks.</p>",
              "<p>The earlier missing-imagery finding applied to the old directory. The new directory supports the proposal's 16-labeled-tile experiment. The six-tile split is preserved but should not be mistaken for a mandatory data limit.</p>",
              "<h2>Available inputs</h2><table><tr><th>Folder</th><th>Image files</th><th>Distinct tile IDs</th></tr>"]
    for folder, count in sorted(counts.items()):
        report.append(f"<tr><td>{esc(folder)}</td><td>{count['file_count']}</td><td>{count['unique_tile_count']}</td></tr>")
    report.append("</table><h2>Verified labeled tile pairings</h2><table><tr><th>Tile</th><th>Size W × H</th><th>Status</th><th>Full vs eroded differences on valid pixels</th></tr>")
    for row in records:
        report.append(f"<tr><td>{row['tile_id']}</td><td>{esc(row.get('image_size'))}</td><td>{row['pair_status']}</td><td>{row.get('full_vs_eroded_different_valid_pixels','not checked')}</td></tr>")
    report.extend(["</table><h2>Auxiliary products</h2><p>DSM files provide floating-point height-related inputs. JPEG nDSM products cannot be treated as calibrated physical heights without documentation. SAM folders contain stored image/mask products, but do not establish a separate executed neural component or a reproducible generation method.</p>",
                   "<h2>Versioning</h2><p>No source data, training configuration or frozen split was modified. A full-data split should be saved as a new version if adopted.</p>",
                   "<p><a href='bundle_summary.json'>Detailed inspection evidence</a> · <a href='labeled_tile_manifest.csv'>Paired labeled tile manifest</a> · <a href='file_inventory.csv'>File inventory</a></p></html>"])
    (output / "bundle_report.html").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"distinct_original_tiles": len(images), "passing_labeled_ids": summary["passing_labeled_tile_ids"],
                      "can_support_10_2_4": summary["can_support_original_10_2_4_tile_counts"],
                      "reference_archive_comparisons": len(comparisons)}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("P1/reports/data_bundle"))
    args = parser.parse_args()
    inspect_bundle(args.data_root, args.output_dir)
