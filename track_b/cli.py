from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from pathlib import Path

from .deform import write_deformed_nifti
from .io import read_manifest, resolve_path, write_ranking_csv
from .mind import DescriptorCache, mind_vector
from .metrics import mean_reciprocal_rank_by_dataset, RetrievalLabel
from .rerank import rerank_ranked_candidates


def _open_text(path: str | Path, mode: str = "r"):
    file_path = Path(path)
    if file_path.suffix == ".gz":
        return gzip.open(file_path, mode + "t", newline="")
    return file_path.open(mode, newline="")


def _read_rankings(path: str | Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with _open_text(path, "r") as handle:
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


def _score_rankings(labels_path: str | Path, rankings_path: str | Path) -> dict[str, float]:
    labels: list[RetrievalLabel] = []
    with _open_text(labels_path, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.append(
                RetrievalLabel(
                    query_id=row["query_id"],
                    target_id=row["target_id"],
                    dataset=row.get("dataset", ""),
                )
            )
    rankings = _read_rankings(rankings_path)
    return mean_reciprocal_rank_by_dataset(labels, rankings)


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


def _seed_for(target_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{target_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def cmd_deform_proxy(args: argparse.Namespace) -> None:
    query_entries = {entry.item_id: entry for entry in read_manifest(args.queries, root=args.data_root)}
    gallery_entries = {entry.item_id: entry for entry in read_manifest(args.gallery, root=args.data_root)}

    deformed_gallery_dir = Path(args.deformed_gallery_dir)
    deformed_gallery_dir.mkdir(parents=True, exist_ok=True)
    deformed_paths: dict[str, Path] = {}
    for entry in gallery_entries.values():
        deformed_path = deformed_gallery_dir / f"{entry.item_id}.nii.gz"
        if not deformed_path.exists() or args.force:
            write_deformed_nifti(
                entry.image_path,
                deformed_path,
                seed=_seed_for(entry.item_id, args.seed),
                max_degrees=args.max_degrees,
                translation=args.translation,
                scale_range=(args.scale_min, args.scale_max),
                coarse_size=(args.elastic_coarse, args.elastic_coarse, args.elastic_coarse),
                elastic_magnitude=args.elastic_magnitude,
            )
        deformed_paths[entry.item_id] = deformed_path

    stage1_cache = DescriptorCache(Path(args.cache_dir)) if args.cache_dir else None
    gallery_vectors = []
    gallery_ids = []
    for entry in gallery_entries.values():
        gallery_path = deformed_paths[entry.item_id]
        vector = stage1_cache.get(gallery_path, max_side=args.max_side) if stage1_cache else mind_vector(
            gallery_path,
            max_side=args.max_side,
        )
        gallery_vectors.append(vector)
        gallery_ids.append(entry.item_id)

    stage1_rows = []
    for entry in query_entries.values():
        query_vector = stage1_cache.get(entry.image_path, max_side=args.max_side) if stage1_cache else mind_vector(
            resolve_path(entry.image_path, args.data_root),
            max_side=args.max_side,
        )
        scores = [float(query_vector @ gallery_vector) for gallery_vector in gallery_vectors]
        order = sorted(range(len(gallery_ids)), key=lambda idx: scores[idx], reverse=True)
        stage1_rows.append((entry.item_id, [gallery_ids[idx] for idx in order]))
    write_ranking_csv(stage1_rows, args.stage1_output)

    stage1_rankings = {query_id: ranking for query_id, ranking in stage1_rows}
    stage2_rows = []
    for query_id, ranked_ids in stage1_rankings.items():
        query_entry = query_entries[query_id]
        candidates = []
        for target_id in ranked_ids[: args.top_k]:
            candidates.append((target_id, deformed_paths[target_id]))
        reranked = rerank_ranked_candidates(
            query_entry.image_path,
            candidates,
            top_k=args.top_k,
            max_side=args.max_side,
        )
        remaining = [target_id for target_id in ranked_ids if target_id not in {item.target_id for item in reranked}]
        stage2_rows.append((query_id, [item.target_id for item in reranked] + remaining))
    write_ranking_csv(stage2_rows, args.stage2_output)

    if args.labels:
        stage2_scores = _score_rankings(args.labels, args.stage2_output)
        print(stage2_scores)


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

    deform = subparsers.add_parser(
        "deform-proxy",
        help="Apply synthetic rigid + elastic deformation to the gallery and rerun stage1+stage2",
    )
    deform.add_argument("--queries", required=True)
    deform.add_argument("--gallery", required=True)
    deform.add_argument("--stage1-output", required=True)
    deform.add_argument("--stage2-output", required=True)
    deform.add_argument("--labels", default=None)
    deform.add_argument("--data-root", default=None)
    deform.add_argument("--deformed-gallery-dir", required=True)
    deform.add_argument("--cache-dir", default=None)
    deform.add_argument("--top-k", type=int, default=10)
    deform.add_argument("--max-side", type=int, default=96)
    deform.add_argument("--seed", type=int, default=20260627)
    deform.add_argument("--max-degrees", type=float, default=12.0)
    deform.add_argument("--translation", type=float, default=0.08)
    deform.add_argument("--scale-min", type=float, default=0.92)
    deform.add_argument("--scale-max", type=float, default=1.08)
    deform.add_argument("--elastic-coarse", type=int, default=6)
    deform.add_argument("--elastic-magnitude", type=float, default=0.12)
    deform.add_argument("--force", action="store_true")
    deform.set_defaults(func=cmd_deform_proxy)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
