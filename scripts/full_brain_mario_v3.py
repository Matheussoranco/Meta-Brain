"""Local/remote CLI for the same restartable experiment shipped in the Colab notebook."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_brain.colab_mario import fetch_data, launch
from meta_brain.mario_trainable import FullBrainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/connectome")
    parser.add_argument("--out-dir", default="results/full_brain_mario_v3/real_seed123")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--bptt", type=int, default=16)
    parser.add_argument("--vision", choices=("real", "zero", "clock"), default="real")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.download:
        fetch_data(args.data_dir)
    cfg = FullBrainConfig(updates=args.updates, max_hours=args.hours, n_envs=args.envs,
                          bptt=args.bptt, vision=args.vision, seed=args.seed)
    results = launch(args.data_dir, str(Path(args.data_dir) / "cache"), args.out_dir, cfg)
    print(results["claim_status"])


if __name__ == "__main__":
    main()
