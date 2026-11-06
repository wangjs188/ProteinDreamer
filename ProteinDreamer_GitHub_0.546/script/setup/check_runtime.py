#!/usr/bin/env python3
"""Runtime versions, imports and optional CUDA check."""
import argparse
import importlib
import importlib.metadata as metadata
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--include-cache-builder", action="store_true",
                        help="Also check optional SiteRM cache-building dependencies.")
    args = parser.parse_args()
    errors = []
    print("Python:", platform.python_version(), "| System:", platform.system(), platform.machine())
    if sys.version_info[:2] != (3, 10):
        errors.append("Use Python 3.10.")
    requirement_files = [ROOT / "requirements.runtime.txt"]
    if args.include_cache_builder:
        requirement_files.append(ROOT / "requirements.cache-build.txt")
    for line in "\n".join(path.read_text() for path in requirement_files).splitlines():
        if "==" not in line or line.startswith(("#", "--")):
            continue
        name, expected = line.split("==", 1)
        try:
            actual = metadata.version(name)
            print(f"{name}: {actual}")
            if actual != expected:
                errors.append(f"{name}: expected {expected}, got {actual}")
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {name}")
    modules = ["torch", "torch_geometric", "torch_scatter", "transformers", "sklearn.metrics",
               "src.saprot", "src.saprot_dataset", "src.world_model.zeroshot_world_model"]
    if args.include_cache_builder:
        modules.extend(["cherryml", "biotite.sequence.phylo", "biotite.structure.io.pdb", "biotraj"])
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"Import {name}: {exc}")
    if args.include_cache_builder:
        try:
            from cherryml.phylogeny_estimation import _fast_cherries
            binary = Path(_fast_cherries.__file__).parent / "FastCherries/build/fast_cherries"
            with binary.open("rb") as handle:
                header = handle.read(20)
            if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 62:
                errors.append("FastCherries must be rebuilt for Linux x86_64; see docs/REPRODUCE_0546.md.")
        except Exception as exc:
            errors.append(f"FastCherries binary check: {exc}")
    try:
        import torch
        print("PyTorch CUDA runtime:", torch.version.cuda, "| GPU visible:", torch.cuda.is_available())
        if torch.version.cuda != "12.1":
            errors.append("Expected the CUDA 12.1 PyTorch wheel.")
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("GPU memory MiB:", torch.cuda.get_device_properties(0).total_memory // 2**20)
            x = torch.ones(2, device="cuda")
            assert float(x.sum()) == 2
        elif args.require_cuda:
            errors.append("CUDA unavailable: use a GPU allocation or check your NVIDIA driver.")
    except Exception as exc:
        errors.append(f"PyTorch check: {exc}")
    if errors:
        for error in errors:
            print("[FAIL]", error)
        raise SystemExit(1)
    print("[PASS] Runtime versions and imports (not a benchmark reproduction).")


if __name__ == "__main__":
    main()
