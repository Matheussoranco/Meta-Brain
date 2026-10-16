"""Build (or reuse) the cached connectome and print the edge-count audit.

    python scripts/build_graph_cache.py --data-dir data/connectome
    python scripts/build_graph_cache.py --data-dir data/connectome --min-weight 5 --rebuild

The report accounts for every edge row read: retained, dropped (unknown or
non-``--status`` bodies, self-loops, pairs below ``--min-weight``) or merged
as a repeated pair. It is also written next to the cache as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_brain.graph import load_connectome  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/connectome")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--min-weight", type=float, default=1.0)
    p.add_argument("--keep-self-loops", action="store_true")
    p.add_argument("--status", nargs="*", default=["Traced"],
                   help="Body statuses to keep (pass --status with no value to keep all).")
    p.add_argument("--rebuild", action="store_true", help="Ignore an existing cache.")
    args = p.parse_args()

    graph = load_connectome(args.data_dir, args.cache_dir, args.min_weight,
                            drop_self_loops=not args.keep_self_loops, statuses=args.status or None,
                            use_cache=not args.rebuild)
    print(f"sensory={len(graph.sensory_indices()):,} motor={len(graph.motor_indices()):,} "
          f"descending={len(graph.descending_indices()):,}")
    print(json.dumps({k: v for k, v in asdict(graph.report).items()
                      if k in ("columns_used", "nt_counts")}, indent=2))


if __name__ == "__main__":
    main()
