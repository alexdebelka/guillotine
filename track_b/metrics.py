from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RetrievalLabel:
    query_id: str
    target_id: str
    dataset: str


def _open_text(path: str | Path, mode: str = "r"):
    file_path = Path(path)
    if file_path.suffix == ".gz":
        return gzip.open(file_path, mode + "t", newline="")
    return file_path.open(mode, newline="")


def reciprocal_rank(ranking: list[str], target_id: str) -> float:
    try:
        index = ranking.index(target_id)
    except ValueError:
        return 0.0
    return 1.0 / float(index + 1)


def mean_reciprocal_rank_by_dataset(labels: Iterable[RetrievalLabel], rankings: dict[str, list[str]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for label in labels:
        rr = reciprocal_rank(rankings.get(label.query_id, []), label.target_id)
        sums[label.dataset] = sums.get(label.dataset, 0.0) + rr
        counts[label.dataset] = counts.get(label.dataset, 0) + 1
    mrrs = {dataset: sums[dataset] / counts[dataset] for dataset in counts if counts[dataset] > 0}
    if not mrrs:
        return {"macro_mrr": 0.0}
    mrrs["macro_mrr"] = sum(mrrs.values()) / float(len(mrrs))
    return mrrs


def load_labels_csv(path: str | Path) -> list[RetrievalLabel]:
    labels: list[RetrievalLabel] = []
    with _open_text(path, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.append(
                RetrievalLabel(
                    query_id=row["query_id"],
                    target_id=row["target_id"],
                    dataset=row.get("dataset", ""),
                )
            )
    return labels


def load_rankings_csv(path: str | Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with _open_text(path, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rankings[row["query_id"]] = row["target_id_ranking"].split()
    return rankings


def score_prediction_file(labels_csv: str | Path, rankings_csv: str | Path) -> dict[str, float]:
    labels = load_labels_csv(labels_csv)
    rankings = load_rankings_csv(rankings_csv)
    return mean_reciprocal_rank_by_dataset(labels, rankings)
