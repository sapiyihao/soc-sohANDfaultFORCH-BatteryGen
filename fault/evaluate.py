"""加载已训练权重，在车辆级和窗口级评估故障分类模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from longseq.data import CLASS_NAMES, LongSequenceDataset
from longseq.model import (
    ConvStemFAFMamba,
    ConvStemMamba,
    LongSequenceMamba,
    MultiScaleHierarchicalMamba,
)
from train import evaluate


MODEL_CLASSES = {
    "baseline": LongSequenceMamba,
    "conv_stem": ConvStemMamba,
    "conv_stem_faf_add": ConvStemFAFMamba,
    "multiscale_hierarchical": MultiScaleHierarchicalMamba,
}


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=root / "data" / "processed_v2")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "experiments" / "current_best_lfp" / "best.pt",
    )
    parser.add_argument("--chemistry", choices=["LFP", "NCM"], default="LFP")
    parser.add_argument("--model", choices=sorted(MODEL_CLASSES), default="multiscale_hierarchical")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, help="仅用于快速运行检查")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size必须大于0，num-workers不能为负")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    dataset = LongSequenceDataset(args.cache, "test", args.chemistry)
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": amp,
    }
    if args.num_workers > 0:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, **loader_options)
    model = MODEL_CLASSES[args.model]().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics, matrix = evaluate(model, loader, device, amp, args.max_batches)
    result = {
        "model": args.model,
        "chemistry": args.chemistry,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "class_names": CLASS_NAMES,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
