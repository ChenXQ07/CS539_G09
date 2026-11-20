"""Read-only inventory and quality audit for a local Vaihingen data directory.

Run from the repository root:
    python -m P1.audit_data --data-root PATH --output-dir P1/reports/data_audit

Requires numpy and Pillow. Original data are never extracted over or modified.
Pairing and label-format checks do not establish annotation provenance or spatial
alignment. No training/validation/test partition is chosen by this audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import platform
import re
import struct
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

CLASSES = {
    0: ("impervious_surfaces", (255, 255, 255)),
    1: ("buildings", (0, 0, 255)),
    2: ("low_vegetation", (0, 255, 255)),
    3: ("trees", (0, 255, 0)),
    4: ("cars", (255, 255, 0)),
    5: ("clutter_background", (255, 0, 0)),
}
IGNORE_RGB = (0, 0, 0)
EXPECTED_LABELED_IDS = (1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37)
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def tile_id(name: str) -> int | None:
    match = re.search(r"(?:area|ISPRS_merge_|V_merge_)(\d+)", name, re.I)
    return int(match.group(1)) if match else None


def role(relative_path: str) -> str:
    folder = relative_path.split("/")[0]
    return {
        "top": "original_tile_candidate",
        "gts_for_participants": "full_reference_candidate",
        "gts_eroded_for_participants": "eroded_reference_candidate",
        "gts_noise_forparticipants": "unverified_noise_label",
        "predict": "prediction_not_ground_truth",
        "dsm": "height_data_out_of_scope",
        "ndsm": "height_data_out_of_scope",
        "ISPRS Vaihingen (nDSM from Markus Gerke)": "height_data_out_of_scope",
    }.get(folder, "derived_data_not_approved_for_training")


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tiff_header(path: Path) -> dict:
    """Inspect a classic TIFF header without disabling Pillow's pixel limit."""
    with path.open("rb") as source:
        first = source.read(8)
        if len(first) != 8 or first[:2] not in (b"II", b"MM"):
            raise ValueError("Not a classic TIFF")
        endian = "<" if first[:2] == b"II" else ">"
        if struct.unpack(endian + "H", first[2:4])[0] != 42:
            raise ValueError("This header reader supports classic TIFF only")
        source.seek(struct.unpack(endian + "I", first[4:8])[0])
        count_data = source.read(2)
        if len(count_data) != 2:
            raise ValueError("Truncated TIFF directory")
        count = struct.unpack(endian + "H", count_data)[0]
        if count > 4096:
            raise ValueError("Unexpectedly large TIFF directory")
        entries = source.read(count * 12)
        if len(entries) != count * 12:
            raise ValueError("Truncated TIFF entries")
        result = {}
        wanted = {256: "width", 257: "height", 258: "bits_per_sample", 277: "channels",
                  33550: "pixel_scale", 33922: "tie_points", 34735: "geo_keys"}
        formats = {1: "B", 3: "H", 4: "I", 12: "d"}
        for offset in range(0, len(entries), 12):
            tag, kind, length = struct.unpack(endian + "HHI", entries[offset:offset + 8])
            if tag not in wanted or kind not in formats or length > 2048:
                continue
            size = struct.calcsize(formats[kind]) * length
            raw = entries[offset + 8:offset + 12]
            if size > 4:
                source.seek(struct.unpack(endian + "I", raw)[0])
                raw = source.read(size)
            else:
                raw = raw[:size]
            values = struct.unpack(endian + str(length) + formats[kind], raw)
            result[wanted[tag]] = values[0] if length == 1 else list(values)
        result["pixel_scan"] = "not_performed_for_large_mosaic"
        return result


def metadata(path: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(path) as image:
                result = {"width": image.width, "height": image.height,
                          "mode": image.mode, "channels": len(image.getbands())}
                tags = getattr(image, "tag_v2", {})
                for tag, field in [(258, "bits_per_sample"), (33550, "pixel_scale"),
                                   (33922, "tie_points"), (34735, "geo_keys")]:
                    if tag in tags:
                        result[field] = list(tags[tag]) if isinstance(tags[tag], tuple) else tags[tag]
                return result
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            return tiff_header(path)


def label_statistics(image: Image.Image) -> dict:
    """Validate exact reference colors. Unknown colors are errors, not class 0."""
    if image.mode != "RGB":
        raise ValueError(f"Expected RGB color reference, received {image.mode}")
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected a three-channel uint8 label")
    packed = rgb[:, :, 0].astype(np.uint32) << 16
    packed |= rgb[:, :, 1].astype(np.uint32) << 8
    packed |= rgb[:, :, 2].astype(np.uint32)
    valid = np.zeros(rgb.shape[:2], dtype=bool)
    counts = {}
    for number, (_, color) in CLASSES.items():
        code = color[0] << 16 | color[1] << 8 | color[2]
        mask = packed == code
        counts[str(number)] = int(np.count_nonzero(mask))
        valid |= mask
    ignored = packed == 0
    unknown = ~(valid | ignored)
    unknown_codes = np.unique(packed[unknown])[:16]
    return {
        "class_pixels": counts, "ignored_pixels": int(np.count_nonzero(ignored)),
        "unknown_pixels": int(np.count_nonzero(unknown)),
        "unknown_color_examples": [[int(c >> 16), int((c >> 8) & 255), int(c & 255)] for c in unknown_codes],
        "total_pixels": int(packed.size),
        "ignore_fraction": float(np.count_nonzero(ignored) / packed.size),
        "palette_check": "pass" if not np.any(unknown) else "fail",
    }


def pairing_checks(image: dict, label: dict) -> list[str]:
    problems = []
    if (image.get("width"), image.get("height")) != (label.get("width"), label.get("height")):
        problems.append("image_label_size_mismatch")
    if image.get("mode") != "RGB" or image.get("dtype") != "uint8":
        problems.append("image_not_three_channel_uint8")
    if label.get("palette_check") != "pass":
        problems.append("label_palette_invalid")
    if label.get("total_pixels", 0) == label.get("ignored_pixels", -1):
        problems.append("label_contains_no_valid_pixels")
    if not image.get("decode_ok") or not label.get("decode_ok"):
        problems.append("pixel_decode_failed")
    return problems


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items() if key in fields})


def preview_pair(image_path: Path, reference: bytes, output: Path, number: int) -> dict:
    """Lossless PNG viewing copies only; never use these thumbnails for training."""
    result = {}
    for kind, source in [("image", image_path), ("label", io.BytesIO(reference))]:
        with Image.open(source) as image:
            image.load()
            image.thumbnail((420, 420), Image.Resampling.NEAREST if kind == "label" else Image.Resampling.BILINEAR)
            name = f"tile_{number}_{kind}.png"
            image.save(output / "previews" / name)
            result[kind] = f"previews/{name}"
    return result


def audit(data_root: Path, output: Path, make_previews: bool = True) -> dict:
    data_root = data_root.resolve()
    output = output.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if output == data_root or data_root in output.parents:
        raise ValueError("Write audit output outside the original data directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    inventory, labels, image_rows, noise_rows = [], {}, {}, []
    problems = []
    files = sorted(p for p in data_root.rglob("*") if p.is_file())
    for path in files:
        relative = path.relative_to(data_root).as_posix()
        record = {"path": relative, "role": role(relative), "tile_id": tile_id(path.name),
                  "size_bytes": path.stat().st_size}
        if path.suffix.lower() in IMAGE_SUFFIXES:
            try:
                record.update(metadata(path))
                record["header_ok"] = True
            except Exception as error:
                record.update(header_ok=False, error=f"{type(error).__name__}: {error}")
                problems.append({"file": relative, "issue": record["error"]})
            if record["role"] == "original_tile_candidate":
                print("Checking original tile:", relative, flush=True)
                try:
                    with Image.open(path) as image:
                        array = np.asarray(image)
                        record.update(dtype=str(array.dtype), decode_ok=True,
                                      pixel_min=int(array.min()), pixel_max=int(array.max()),
                                      sha256=sha256_file(path))
                except Exception as error:
                    record.update(decode_ok=False, error=f"{type(error).__name__}: {error}")
                number = record["tile_id"]
                if number in image_rows:
                    raise ValueError(f"Duplicate original tile ID: {number}")
                image_rows[number] = record
            elif record["role"] in {"eroded_reference_candidate", "full_reference_candidate", "unverified_noise_label"}:
                try:
                    with Image.open(path) as image:
                        record.update(label_statistics(image), decode_ok=True, sha256=sha256_file(path))
                except Exception as error:
                    record.update(decode_ok=False, error=f"{type(error).__name__}: {error}")
                if record["role"] == "unverified_noise_label":
                    noise_rows.append(record)
        inventory.append(record)

    # Verify every archive member's CRC while reading, not just its filename.
    archive_details = []
    for archive in sorted(data_root.glob("*.zip")):
        if "eroded" not in archive.name.lower():
            continue
        print("Checking reference archive:", archive.name, flush=True)
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                if entry.is_dir() or not re.fullmatch(r"top_mosaic_09cm_area\d+_noBoundary\.tif", Path(entry.filename).name):
                    continue
                number = tile_id(entry.filename)
                record = {"tile_id": number, "path": f"{archive.name}!{entry.filename}",
                          "archive": archive.name, "entry": entry.filename, "size_bytes": entry.file_size,
                          "role": "archived_eroded_reference_candidate", "crc_ok": False}
                print("Checking reference tile:", number, flush=True)
                try:
                    raw = source.read(entry)
                    record["crc_ok"] = True
                    with Image.open(io.BytesIO(raw)) as image:
                        record.update(width=image.width, height=image.height, mode=image.mode,
                                      channels=len(image.getbands()), **label_statistics(image))
                    record.update(decode_ok=True, crc_ok=True, sha256=hashlib.sha256(raw).hexdigest())
                except Exception as error:
                    record.update(decode_ok=False, error=f"{type(error).__name__}: {error}")
                if number in labels:
                    raise ValueError(f"Duplicate archived reference ID {number}; choose a canonical archive")
                labels[number] = record
                archive_details.append(record)

    rows, previews = [], []
    for number in sorted(set(EXPECTED_LABELED_IDS) | set(image_rows) | set(labels)):
        image, label = image_rows.get(number), labels.get(number)
        row = {"tile_id": number, "in_planned_labeled_set": number in EXPECTED_LABELED_IDS,
               "image_path": image["path"] if image else "", "reference_path": label["path"] if label else "",
               "image_width": image.get("width", "") if image else "", "image_height": image.get("height", "") if image else "",
               "reference_width": label.get("width", "") if label else "", "reference_height": label.get("height", "") if label else "",
               "image_sha256": image.get("sha256", "") if image else "", "reference_sha256": label.get("sha256", "") if label else "",
               "class_pixels": label.get("class_pixels", {}) if label else {},
               "ignored_pixels": label.get("ignored_pixels", "") if label else "",
               "unknown_pixels": label.get("unknown_pixels", "") if label else "",
               "ignore_fraction": label.get("ignore_fraction", "") if label else "",
               "provenance_verified": False, "spatial_alignment_review": "pending",
               "split": "unassigned"}
        checks = []
        if not image:
            checks.append("missing_original_tile")
        if not label:
            checks.append("no_archived_reference_for_this_id")
        if image and label:
            checks.extend(pairing_checks(image, label))
        row["issues"] = checks
        row["pair_checks"] = "pass" if not checks else "not_ready"
        rows.append(row)
        if make_previews and not checks:
            with zipfile.ZipFile(data_root / label["archive"]) as source:
                previews.append({"tile_id": number, **preview_pair(data_root / image["path"], source.read(label["entry"]), output, number)})

    comparisons = []
    for record in noise_rows:
        label = labels.get(record["tile_id"])
        if not label or not label.get("decode_ok") or not record.get("decode_ok"):
            continue
        with zipfile.ZipFile(data_root / label["archive"]) as source:
            with Image.open(io.BytesIO(source.read(label["entry"]))) as image:
                reference = np.array(image)
        with Image.open(data_root / record["path"]) as image:
            candidate = np.array(image)
        comp = {"tile_id": record["tile_id"], "file": record["path"], "promoted_to_ground_truth": False}
        if candidate.shape != reference.shape:
            comp["comparison"] = "different_dimensions"
        else:
            valid = np.any(reference != 0, axis=-1)
            differing = valid & np.any(candidate != reference, axis=-1)
            comp.update(comparison="compared_on_nonblack_reference_pixels", compared_pixels=int(valid.sum()),
                        different_pixels=int(differing.sum()),
                        difference_fraction=float(differing.sum() / valid.sum()) if valid.any() else None)
        comparisons.append(comp)

    local_reference_checks = []
    for record in inventory:
        if record["role"] == "eroded_reference_candidate" and record["tile_id"] in labels:
            local_reference_checks.append({"file": record["path"], "tile_id": record["tile_id"],
                                           "identical_bytes_to_archive": record.get("sha256") == labels[record["tile_id"]].get("sha256")})

    expected_missing = sorted(set(EXPECTED_LABELED_IDS) - set(image_rows))
    passed = [r["tile_id"] for r in rows if r["pair_checks"] == "pass"]
    mosaic = next((r for r in inventory if r["path"] == "TOP_Mosaic_09cm.tif"), None)
    summary = {
        "audit_scope": "Local file checks only; no downloads, no source edits, no splits assigned",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "Pillow": pillow_version},
        "source_directory_name": data_root.name,
        "files_in_inventory": len(inventory), "original_tile_ids": sorted(image_rows),
        "planned_labeled_tile_ids": list(EXPECTED_LABELED_IDS),
        "archived_eroded_reference_ids": sorted(labels), "structurally_matched_pair_ids": passed,
        "missing_original_tiles_for_planned_labeled_set": expected_missing,
        "original_tiles_without_archived_reference": sorted(set(image_rows) - set(labels)),
        "full_reference_directory_exists": (data_root / "gts_for_participants").is_dir(),
        "archived_references_all_decode_and_palette_valid": bool(labels) and all(r.get("decode_ok") and r.get("palette_check") == "pass" for r in labels.values()),
        "ready_for_planned_10_2_4_split": len(passed) >= 16 and not expected_missing,
        "ready_for_training": False,
        "training_readiness_reason": "Source/provenance and spatial alignment need confirmation; missing tiles must be resolved before the planned split.",
        "zip_checks": archive_details, "local_reference_byte_checks": local_reference_checks,
        "noise_label_comparisons": comparisons,
        "large_mosaic_header": mosaic,
        "mosaic_note": "A full mosaic is not automatically a replacement for missing tiles. Matching tile origins/extents are needed; none are inferred from labels.",
        "issues": problems,
        "label_contract": {"classes": {str(k): {"name": v[0], "rgb": list(v[1])} for k, v in CLASSES.items()},
                           "ignore_rgb": list(IGNORE_RGB), "planned_ignore_index": 255,
                           "unknown_color_policy": "flag for investigation; do not silently map to class zero"},
        "limitations": ["File names, valid palettes and matching dimensions do not prove label provenance or spatial correctness.",
                        "The large mosaic receives header checks only, not full pixel decompression.",
                        "Derived, noisy and predicted labels are inventoried but not approved as ground truth."]}
    write_csv(output / "file_inventory.csv", inventory, ["path", "role", "tile_id", "size_bytes", "width", "height", "mode", "channels", "bits_per_sample", "header_ok", "dtype", "decode_ok", "sha256", "error"])
    write_csv(output / "data_manifest.csv", rows, list(rows[0]))
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    create_html(output, summary, rows, previews)
    print(json.dumps({k: summary[k] for k in ["original_tile_ids", "structurally_matched_pair_ids", "missing_original_tiles_for_planned_labeled_set", "ready_for_planned_10_2_4_split"]}), flush=True)
    return summary


def create_html(output: Path, summary: dict, rows: list[dict], previews: list[dict]) -> None:
    escape = lambda value: html.escape(str(value))
    body = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Vaihingen data audit</title>",
            "<style>body{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 22px;color:#193047}h1,h2{color:#16324e}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #ccd5de;padding:9px;text-align:left}th{background:#eef3f7}figure{margin:20px 0 40px}figure div{display:flex;gap:16px;align-items:flex-start}img{max-width:42%;height:auto;border:1px solid #ccd5de}.status{padding:16px;background:#fff4df}code{background:#eef3f7}small{color:#526373}</style>",
            "<h1>Vaihingen data inventory and quality audit</h1>",
            "<p>Read-only checks of local files. No source data were modified and no tile partitions were assigned.</p>",
            "<p class='status'><strong>Training readiness: pending.</strong> Passing file checks does not verify provenance or spatial alignment.</p>",
            f"<p>Original tile IDs: {escape(summary['original_tile_ids'])}. Structurally matched pairs: {escape(summary['structurally_matched_pair_ids'])}.</p>",
            f"<p><strong>Missing originals for the proposed 16-tile labeled subset:</strong> {escape(summary['missing_original_tiles_for_planned_labeled_set'])}.</p>",
            "<h2>Pairing checks</h2><table><tr><th>Tile</th><th>Image size (W × H)</th><th>Reference size</th><th>Checks</th><th>Ignored pixels</th></tr>"]
    for row in rows:
        ratio = row["ignore_fraction"]
        ratio = f"{ratio:.2%}" if isinstance(ratio, float) else "—"
        body.append(f"<tr><td>{row['tile_id']}</td><td>{row['image_width']} × {row['image_height']}</td><td>{row['reference_width']} × {row['reference_height']}</td><td>{escape(', '.join(row['issues']) or 'file checks passed; review pending')}</td><td>{ratio}</td></tr>")
    body.append("</table><h2>Image and reference viewing copies</h2><p>Left: IRRG imagery displayed in channel order. Right: eroded reference. Black reference pixels are excluded boundaries or undefined pixels. Viewing copies are downsampled and must not be used as training data.</p>")
    for preview in previews:
        body.append(f"<figure><figcaption><strong>Tile {preview['tile_id']}</strong></figcaption><div><img alt='IRRG tile' src='{escape(preview['image'])}'><img alt='Reference labels' src='{escape(preview['label'])}'></div></figure>")
    body.append("<h2>Noise-label comparison</h2><p>Only nonblack archived reference pixels are compared. Matching some pixels does not establish trustworthy labels for the excluded boundaries.</p><table><tr><th>Tile</th><th>Different reference pixels</th><th>Fraction</th></tr>")
    for comparison in summary["noise_label_comparisons"]:
        fraction = comparison.get("difference_fraction")
        body.append(f"<tr><td>{comparison['tile_id']}</td><td>{comparison.get('different_pixels', 'not comparable')}</td><td>{fraction:.2%}</td></tr>" if fraction is not None else f"<tr><td>{comparison['tile_id']}</td><td colspan='2'>Not comparable</td></tr>")
    body.append("</table><h2>Limitations and next actions</h2><ul><li>Obtain the missing original tiles from the team or the verified dataset source.</li><li>Confirm reference provenance and review alignment before defining the fixed 10/2/4 split.</li><li>Do not treat files under noise, predict or SAM folders as approved reference labels.</li><li>The large mosaic requires known tile extents before it could supply missing tile crops.</li></ul><p>Machine-readable evidence: <a href='audit_summary.json'>audit_summary.json</a>, <a href='data_manifest.csv'>data_manifest.csv</a>, <a href='file_inventory.csv'>file_inventory.csv</a>.</p></html>")
    (output / "quality_report.html").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("P1/reports/data_audit"))
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    audit(args.data_root, args.output_dir, not args.no_previews)


if __name__ == "__main__":
    main()
