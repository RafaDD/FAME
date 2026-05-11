from __future__ import annotations


def corr_metrics(x: list, y: list) -> dict:
    import numpy as np

    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    n = len(x_arr)
    if n < 2:
        return {"pearson_r": float("nan"), "spearman_r": float("nan")}

    def _rankdata(a: "np.ndarray") -> "np.ndarray":
        """Average-rank ties."""
        sorted_idx = np.argsort(a, kind="stable")
        ranks = np.empty(len(a), dtype=float)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[sorted_idx[j]] == a[sorted_idx[i]]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg_rank
            i = j
        return ranks

    pearson_r = float(np.corrcoef(x_arr, y_arr)[0, 1])
    rx = _rankdata(x_arr)
    ry = _rankdata(y_arr)
    spearman_r = float(np.corrcoef(rx, ry)[0, 1])

    return {"pearson_r": pearson_r, "spearman_r": spearman_r}
