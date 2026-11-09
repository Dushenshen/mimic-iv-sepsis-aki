from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parents[1]
base = root / "rebuild20261026"
sys.path.insert(0, str(base))
from analyze import CORE, prepare

out = Path(__file__).with_name("matched_cv_20261107")
out.mkdir(exist_ok=True)
d = prepare(pd.read_csv(base / "outputs/primary_reconstructed.csv", parse_dates=["intime"]))
y = d.death28.to_numpy()
assert len(d) == d.subject_id.nunique() == 631 and y.sum() == 226
labs = pd.read_csv(base / "outputs/creatinine_observations.csv", parse_dates=["charttime"])
labs = labs.merge(d[["stay_id", "intime"]], on="stay_id", validate="many_to_one")
h = (labs.charttime - labs.intime).dt.total_seconds() / 3600
labs = labs.assign(day=h // 24).loc[h.ge(0) & h.lt(72)].sort_values(["stay_id", "charttime"], kind="stable")
daily = labs.groupby(["stay_id", "day"]).creatinine.agg(["max", "last", "median"]).unstack().reindex(d.stay_id)
np.testing.assert_allclose(d[["cr_d0", "cr_d2"]], daily[[("max", 0), ("max", 2)]])
x = d[CORE + ["circ_unimproved"]].to_numpy()
designs = {"Base": x}
for summary in ("max", "last", "median"):
    a, b = daily[(summary, 0)].to_numpy(), daily[(summary, 2)].to_numpy()
    assert np.isfinite(a).all() and np.isfinite(b).all() and min(a.min(), b.min()) > 0
    designs[summary + "/Binary"] = np.column_stack((x, b >= .8*a - 1e-8))
    designs[summary + "/Continuous"] = np.column_stack((x, np.log(b/a)))
pred = {key: np.full((10, len(d)), np.nan) for key in designs}
for r in range(10):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=20261028+r).split(x, y):
        for key, a in designs.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=3000, tol=1e-9))
            pred[key][r, te] = model.fit(a[tr], y[tr]).predict_proba(a[te])[:, 1]
assert all(np.isfinite(p).all() for p in pred.values())
def score(p, idx):
    yy = y[idx]
    pp = p[:, idx]
    return np.mean([roc_auc_score(yy, row) for row in pp]), np.mean((pp-yy)**2)

idx = np.arange(len(y))
observed = {key: score(p, idx) for key, p in pred.items()}
rng = np.random.default_rng(20261107)
groups = [idx[y == i] for i in (0, 1)]
boot = {key: [] for key in designs if key != "Base"}
for _ in range(1000):
    draw = np.concatenate([rng.choice(g, len(g)) for g in groups])
    baseline = np.array(score(pred["Base"], draw))
    for key in boot:
        boot[key].append(np.array(score(pred[key], draw))-baseline)
rows = []
for key, (auc, brier) in observed.items():
    row = dict(model=key, auc=auc, brier=brier)
    if key in boot:
        limits = np.quantile(boot[key], [.025, .975], axis=0)
        row.update(d_auc=auc-observed["Base"][0], d_brier=brier-observed["Base"][1],
                   auc_lo=limits[0, 0], auc_hi=limits[1, 0], brier_lo=limits[0, 1], brier_hi=limits[1, 1])
    rows.append(row)
pd.DataFrame(rows).to_csv(out / "matched_cv_results.csv", index=False)
print(pd.DataFrame(rows).round(5).to_string(index=False))
