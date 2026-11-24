"""Image-level multi-label presence measured against valid semantic GT pixels."""
from __future__ import annotations

import numpy as np


def average_precision(truth, scores):
    """Non-interpolated AP, grouping tied scores into one decision threshold."""
    truth, scores = np.asarray(truth, dtype=bool), np.asarray(scores, dtype=float)
    if not truth.any():
        return None
    order = np.argsort(-scores, kind="stable")
    truth, scores = truth[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = truth.cumsum()[ends]
    increments = np.diff(np.r_[0, tp])
    return float(np.sum(increments * tp / (ends + 1)) / truth.sum())


class PresenceMetrics:
    def __init__(self, classes, *, source):
        self.classes = list(classes)
        self.source = source
        self.counts = np.zeros((len(self.classes), 4), dtype=np.int64)  # TP, FP, FN, TN
        self.images = 0
        self.exact = 0
        self.score_rows = []

    def update(self, target, prediction, valid, *, scores=None, sample_id=None):
        target, prediction, valid = np.asarray(target), np.asarray(prediction), np.asarray(valid, dtype=bool)
        if target.shape != prediction.shape or target.shape != valid.shape:
            raise ValueError("Presence masks must share the original GT grid")
        if not valid.any():
            return
        class_ids = np.array([raw_id for raw_id, _ in self.classes])
        truth = np.isin(class_ids, np.unique(target[valid]))
        if scores is None:
            predicted = np.isin(class_ids, np.unique(prediction[valid]))
        else:
            scores = np.asarray(scores, dtype=float)
            if scores.shape != truth.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Presence scores must be finite class probabilities in [0, 1]")
            predicted = scores >= 0.5
            self.score_rows.append(dict(sample_id=sample_id, target=truth.astype(int).tolist(),
                                        probabilities=scores.tolist()))
        self.counts += np.stack((truth & predicted, ~truth & predicted,
                                 truth & ~predicted, ~truth & ~predicted), axis=1)
        self.images += 1
        self.exact += int(np.array_equal(truth, predicted))

    def result(self):
        tp, fp, fn, tn = self.counts.T.astype(float)
        def ratio(a, b):
            return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        f1 = ratio(2 * tp, 2 * tp + fp + fn)
        supported = tp + fn > 0
        total = self.counts.sum()
        aps = [None] * len(self.classes)
        if self.images and len(self.score_rows) == self.images:
            truth = np.array([row["target"] for row in self.score_rows])
            scores = np.array([row["probabilities"] for row in self.score_rows])
            aps = [average_precision(truth[:, i], scores[:, i]) for i in range(len(self.classes))]
        available_ap = [value for value in aps if value is not None]
        return dict(source=self.source, images=self.images,
                    mean_average_precision=float(np.mean(available_ap)) if available_ap else None,
                    average_precision_policy="non-interpolated; tied scores grouped; classes with positive GT support",
                    score_rows=self.score_rows if self.score_rows else None,
                    threshold=0.5 if self.source == "native_head" else None,
                    spatial_rule="any predicted pixel on valid GT region" if self.source == "segmentation_mask" else "max tile probability",
                    accuracy=float((tp + tn).sum() / total) if total else None,
                    exact_match_accuracy=self.exact / self.images if self.images else None,
                    micro_precision=float(ratio(tp.sum(), (tp + fp).sum())) if total else None,
                    micro_recall=float(ratio(tp.sum(), (tp + fn).sum())) if total else None,
                    micro_f1=float(ratio(2 * tp.sum(), (2 * tp + fp + fn).sum())) if total else None,
                    macro_f1=float(f1[supported].mean()) if supported.any() else None,
                    macro_policy="classes with positive GT support; zero predicted positives gives F1=0",
                    per_class=[dict(class_id=raw_id, class_name=name, tp=int(tp[i]), fp=int(fp[i]),
                                    fn=int(fn[i]), tn=int(tn[i]), precision=float(precision[i]),
                                    recall=float(recall[i]), f1=float(f1[i]), average_precision=aps[i])
                               for i, (raw_id, name) in enumerate(self.classes)])
