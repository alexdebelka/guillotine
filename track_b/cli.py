from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .io import read_manifest, resolve_path, write_ranking_csv
from .mind import DescriptorCache, mind_vector
from .rerank import rerank_ranked_candidates


def _read_rankings(path: str | Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rankings[row["query_id"]] = row["target_id_ranking"].split()
    return rankings


def cmd_rank(args: argparse.Namespace) -> None:
    query_entries = read_manifest(args.queries, root=args.data_root)
    gallery_entries = read_manifest(args.gallery, root=args.data_root)
    cache = DescriptorCache(Path(args.cache_dir)) if args.cache_dir else None

    gallery_vectors = []
    gallery_ids = []
    for entry in gallery_entries:
        vector = cache.get(entry.image_path, max_side=args.max_side) if cache else mind_vector(
            resolve_path(entry.image_path, args.data_root),
            max_side=args.max_side,
        )
        gallery_vectors.append(vector)
        gallery_ids.append(entry.item_id)

    rows = []
    for entry in query_entries:
        query_vector = cache.get(entry.image_path, max_side=args.max_side) if cache else mind_vector(
            resolve_path(entry.image_path, args.data_root),
            max_side=args.max_side,
        )
        scores = [float(query_vector @ gallery_vector) for gallery_vector in gallery_vectors]
        order = sorted(range(len(gallery_ids)), key=lambda idx: scores[idx], reverse=True)
        rows.append((entry.item_id, [gallery_ids[idx] for idx in order]))
    write_ranking_csv(rows, args.output)


def cmd_rerank(args: argparse.Namespace) -> None:
    query_entries = {entry.item_id: entry for entry in read_manifest(args.queries, root=args.data_root)}
    gallery_entries = {entry.item_id: entry for entry in read_manifest(args.gallery, root=args.data_root)}
    rankings = _read_rankings(args.rankings)
    rows = []
    for query_id, ranked_ids in rankings.items():
        query_entry = query_entries[query_id]
        candidates = []
        for target_id in ranked_ids[: args.top_k]:
            candidate_entry = gallery_entries[target_id]
            candidates.append((target_id, candidate_entry.image_path))
        reranked = rerank_ranked_candidates(
            query_entry.image_path,
            candidates,
            top_k=args.top_k,
            max_side=args.max_side,
        )
        remaining = [target_id for target_id in ranked_ids if target_id not in {item.target_id for item in reranked}]
        rows.append((query_id, [item.target_id for item in reranked] + remaining))
    write_ranking_csv(rows, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="track-b")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank = subparsers.add_parser("rank", help="Rank a gallery with MIND-SSC descriptors")
    rank.add_argument("--queries", required=True)
    rank.add_argument("--gallery", required=True)
    rank.add_argument("--output", required=True)
    rank.add_argument("--data-root", default=None)
    rank.add_argument("--cache-dir", default=None)
    rank.add_argument("--max-side", type=int, default=96)
    rank.set_defaults(func=cmd_rank)

    rerank = subparsers.add_parser("rerank", help="Re-rank the top-k with label-free geometry")
    rerank.add_argument("--queries", required=True)
    rerank.add_argument("--gallery", required=True)
    rerank.add_argument("--rankings", required=True)
    rerank.add_argument("--output", required=True)
    rerank.add_argument("--data-root", default=None)
    rerank.add_argument("--top-k", type=int, default=10)
    rerank.add_argument("--max-side", type=int, default=96)
    rerank.set_defaults(func=cmd_rerank)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
