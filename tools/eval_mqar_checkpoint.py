#!/usr/bin/env python3
"""Evaluate a checkpoint produced by benchmark_mqar_fair.py."""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "experiments"))

import torch

from benchmark_mqar_fair import build_model, recall_at_distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = SimpleNamespace(**saved["args"])
    model = build_model(model_args, device)
    model.load_state_dict(saved["model"])
    recall = recall_at_distances(model, args.data_seed, model_args.m, device,
                                 args.batch_size, args.repeats)
    print(json.dumps({"checkpoint": args.checkpoint, "selected_step": saved["step"],
                      "data_seed": args.data_seed, **recall}, sort_keys=True))


if __name__ == "__main__":
    main()
