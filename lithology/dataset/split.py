"""Group-by-well train/val/test split (spec section 14).

No depth point from the same well is ever allowed to appear in more than
one split -- splitting happens on the list of well ids, never on rows.
When there are too few wells for a stable holdout split, grouped k-fold
cross-validation over wells is used instead (still never mixing a well
across folds).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from lithology.config import SplitConfig


class SplitError(Exception):
    pass


@dataclass
class SplitResult:
    mode: str                    # "holdout" | "cross_validation" | "explicit"
    train: Optional[list] = None
    val: Optional[list] = None
    test: Optional[list] = None
    folds: Optional[list] = None  # list of {"train": [...], "val": [...]}

    def as_dict(self) -> dict:
        d = {"mode": self.mode}
        if self.mode in ("holdout", "explicit"):
            d.update(train=self.train, val=self.val, test=self.test)
        else:
            d["folds"] = self.folds
        return d

    def assert_disjoint(self) -> None:
        if self.mode in ("holdout", "explicit"):
            groups = [self.train or [], self.val or [], self.test or []]
            seen = set()
            for g in groups:
                overlap = seen & set(g)
                if overlap:
                    raise SplitError(f"Well(s) {overlap} appear in more than one split.")
                seen |= set(g)
        else:
            for fold in self.folds:
                overlap = set(fold["train"]) & set(fold["val"])
                if overlap:
                    raise SplitError(f"Well(s) {overlap} appear in both train and val of a fold.")


def split_wells(well_ids: list, config: SplitConfig) -> SplitResult:
    well_ids = sorted(set(well_ids))
    n = len(well_ids)
    if n == 0:
        raise SplitError("Cannot split an empty well list.")

    if config.train_wells or config.val_wells or config.test_wells:
        train = list(config.train_wells or [])
        val = list(config.val_wells or [])
        test = list(config.test_wells or [])
        all_ids = set(well_ids)
        explicit_ids = set(train) | set(val) | set(test)
        unknown = explicit_ids - all_ids
        if unknown:
            raise SplitError(f"Explicit split references unknown well id(s): {sorted(unknown)}")
        missing = all_ids - explicit_ids
        result = SplitResult(mode="explicit", train=train, val=val, test=test)
        result.assert_disjoint()
        if missing:
            # Never silently drop a well: park it in train by default and say so.
            result.train = sorted(set(result.train) | missing)
        return result

    rng = random.Random(config.seed)
    shuffled = well_ids[:]
    rng.shuffle(shuffled)

    if n < config.min_wells_for_holdout_split:
        k = max(min(config.n_folds, n), 2)
        fold_bins = [shuffled[i::k] for i in range(k)]
        folds = []
        for i in range(k):
            val_wells = fold_bins[i]
            train_wells = [w for j, b in enumerate(fold_bins) if j != i for w in b]
            folds.append({"train": train_wells, "val": val_wells})
        result = SplitResult(mode="cross_validation", folds=folds)
        result.assert_disjoint()
        return result

    n_train = max(int(round(config.train_frac * n)), 1)
    n_val = max(int(round(config.val_frac * n)), 1) if config.val_frac > 0 else 0
    n_train = min(n_train, n - 1) if n > 1 else n_train
    n_val = min(n_val, max(n - n_train - 1, 0))
    n_test = n - n_train - n_val

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    if n_test == 0 and n_val > 0:
        # borrow one well from val for test so a test split always exists when n allows
        test, val = val[-1:], val[:-1]

    result = SplitResult(mode="holdout", train=train, val=val, test=test)
    result.assert_disjoint()
    return result
