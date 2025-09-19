# tools/clean_vendor.py
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

REL_TARGETS_REQUIRED: List[str] = [
    # strictly unused
    "indextts2/vendor/indextts/infer.py",
    "indextts2/vendor/indextts/infer_v2.py.bak",
    "indextts2/vendor/indextts/cli.py",
    "indextts2/vendor/indextts/vqvae",
]

REL_TARGETS_AGGRESSIVE: List[str] = [
    # optional aggressive cleaning; runtime will fallback without these
    "indextts2/vendor/indextts/BigVGAN/alias_free_activation/cuda",
]

def ensure_under(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False

def human_size(n: int) -> str:
    units = ["B","KB","MB","GB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units)-1:
        x /= 1024
        i += 1
    return f"{x:.2f}{units[i]}"

def collect_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except Exception:
                pass
    return total

def backup_to_zip(root: Path, items: List[Path], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in items:
            if not p.exists():
                continue
            if p.is_file():
                arcname = p.resolve().relative_to(root.resolve()).as_posix()
                zf.write(p, arcname)
            else:
                for f in p.rglob("*"):
                    if f.is_file():
                        arcname = f.resolve().relative_to(root.resolve()).as_posix()
                        zf.write(f, arcname)

def delete_path(p: Path) -> None:
    if not p.exists():
        return
    if p.is_file() or p.is_symlink():
        p.unlink(missing_ok=True)
        return
    shutil.rmtree(p, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Safely prune unused vendored files.")
    parser.add_argument("--root", type=str, default=None, help="Repo root (defaults to this script's parent parent).")
    parser.add_argument("--apply", action="store_true", help="Actually delete files. By default it's a dry-run.")
    parser.add_argument("--backup", action="store_true", help="Zip-backup targets before deleting.")
    parser.add_argument("--backup-path", type=str, default="tools/vendor_prune_backup.zip", help="Zip path.")
    parser.add_argument("--aggressive", action="store_true", help="Also remove optional CUDA kernels.")
    args = parser.parse_args()

    # Resolve repo root: default to script parent parent (tools/clean_vendor.py -> repo)
    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent
    root = Path(args.root).resolve() if args.root else default_root

    print(f"[i] Repo root: {root}")
    if not root.exists():
        print("[!] Root path does not exist.")
        sys.exit(2)

    targets_rel = list(REL_TARGETS_REQUIRED)
    if args.aggressive:
        targets_rel += REL_TARGETS_AGGRESSIVE

    # Build absolute list with safety check
    targets_abs: List[Path] = []
    for rel in targets_rel:
        p = (root / rel).resolve()
        if not ensure_under(root, p):
            print(f"[skip] Unsafe target (outside root): {p}")
            continue
        targets_abs.append(p)

    # Summarize
    to_remove: List[Tuple[Path, int]] = []
    total_bytes = 0
    for p in targets_abs:
        size = collect_size(p)
        status = "MISSING" if not p.exists() else "OK"
        print(f"    - {p.relative_to(root)}  [{status}]  {human_size(size)}")
        if p.exists():
            to_remove.append((p, size))
            total_bytes += size

    print(f"\n[i] Targets existing: {len(to_remove)} items, total size ≈ {human_size(total_bytes)}")
    if not to_remove:
        print("[i] Nothing to remove. Done.")
        return

    if args.backup:
        zip_path = (root / args.backup_path).resolve()
        print(f"[i] Creating backup zip: {zip_path}")
        try:
            backup_to_zip(root, [p for p, _ in to_remove], zip_path)
            print("[i] Backup completed.")
        except Exception as e:
            print(f"[!] Backup failed: {e}")
            sys.exit(3)

    if not args.apply:
        print("\n[DRY-RUN] No deletion performed. Re-run with --apply to delete.")
        return

    # Delete
    failures = 0
    for p, _ in to_remove:
        try:
            delete_path(p)
            print(f"[del] {p.relative_to(root)}")
        except Exception as e:
            failures += 1
            print(f"[!] Failed to delete {p}: {e}")

    if failures == 0:
        print("[i] Deletion completed successfully.")
    else:
        print(f"[!] Deletion completed with {failures} failures.")

if __name__ == "__main__":
    main()