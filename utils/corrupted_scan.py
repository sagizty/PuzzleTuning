# ---- corrupted_scan.py (you can also paste directly into your training script) ----
import os
from typing import Iterable, List, Tuple, Optional
from PIL import Image, ImageFile
from tqdm import tqdm

# Be tolerant (avoid false positives from truncated/large images)
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS_DEFAULT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# --- Worker (must be top-level for multiprocessing) ---
def _verify_one(path: str) -> Optional[Tuple[str, str]]:
    """Return None if ok, else (path, error_string)."""
    try:
        with Image.open(path) as im:
            im.verify()  # header check only
        return None
    except Exception as e:
        return (path, f"{type(e).__name__}: {e}")

def _collect_paths(root_dir: str, exts: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(r, f))
    return out

def scan_corrupted_images(
    root_dir: str,
    out_txt: Optional[str] = "./broken_images.txt",
    exts: Tuple[str, ...] = IMAGE_EXTS_DEFAULT,
    mode: str = "process",                # "process" | "thread"
    num_workers: Optional[int] = None,    # None => auto (CPU-1 for process, 32 for thread)
    show_progress: bool = True,
    strict: bool = True,                  # raise if broken files found
    ddp_barrier: bool = True,             # call barrier after rank0 finishes
) -> List[str]:
    """
    Scan dataset for unreadable images. Returns list of broken file paths.
    If strict=True and any broken files are found, raises RuntimeError.

    Safe to call in DDP: only rank 0 scans; others wait at a barrier.
    """
    # Detect DDP and rank
    rank, world_size, is_ddp = 0, 1, False
    try:
        import torch.distributed as dist
        is_ddp = dist.is_available() and dist.is_initialized()
        if is_ddp:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    except Exception:
        is_ddp = False

    # Only rank 0 performs the scan
    broken_paths: List[str] = []
    if (not is_ddp) or (rank == 0):
        paths = _collect_paths(root_dir, exts)
        total = len(paths)
        print(f"[scan] Found {total:,} image files under: {root_dir}")

        if mode not in ("process", "thread"):
            raise ValueError("mode must be 'process' or 'thread'")

        if num_workers is None:
            if mode == "process":
                import multiprocessing as mp
                num_workers = max(1, (mp.cpu_count() or 1) - 1)
            else:
                # threads work well here (IO-bound PIL verify); cap to avoid oversubscription
                num_workers = 32

        iterator: Iterable[Optional[Tuple[str, str]]]
        if mode == "process":
            import multiprocessing as mp
            # chunksize tuned for throughput
            chunksize = max(32, total // (max(1, num_workers) * 10) if total else 32)
            with mp.Pool(processes=num_workers) as pool:
                iterator = pool.imap_unordered(_verify_one, paths, chunksize=chunksize)
                broken_paths = _consume_with_tqdm(iterator, total, show_progress)
        else:
            # thread pool (no pickling needed)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            broken_paths = []
            pbar = tqdm(total=total, desc="Scanning images", unit="img") if show_progress else None
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(_verify_one, p) for p in paths]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None:
                        broken_paths.append(res[0])
                    if pbar:
                        pbar.update(1)
            if pbar:
                pbar.close()

        if broken_paths:
            print(f"[scan] Broken files: {len(broken_paths):,}")
            for p in broken_paths[:50]:
                print(p)
            if out_txt:
                os.makedirs(os.path.dirname(out_txt), exist_ok=True) if os.path.dirname(out_txt) else None
                with open(out_txt, "w") as f:
                    for p in broken_paths:
                        f.write(p + "\n")
                print(f"[scan] Wrote list → {out_txt}")
            if strict:
                raise RuntimeError(
                    f"{len(broken_paths)} broken files found under {root_dir}. "
                    f"See list at {out_txt or '(no file)'}."
                )
        else:
            print("[scan] No broken files found.")

    # Keep other ranks in sync
    if is_ddp and ddp_barrier:
        import torch.distributed as dist
        dist.barrier()

    return broken_paths

def _consume_with_tqdm(iterator: Iterable[Optional[Tuple[str, str]]], total: int, show: bool) -> List[str]:
    bad: List[str] = []
    bar = tqdm(total=total, desc="Scanning images", unit="img") if show else None
    for res in iterator:
        if res is not None:
            bad.append(res[0])
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    return bad


if __name__ == '__main__':
    scan_corrupted_images(
        root_dir="/root/autodl-tmp/datasets/All",
        out_txt="broken_images.txt",
        mode="process",  # or "thread" if you prefer (no pickling)
        num_workers=None,  # auto
        show_progress=True,
        strict=True,  # raise to stop if broken files found
        ddp_barrier=True,  # keep ranks in sync
    )
