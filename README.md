# Early creatinine change and circulatory status

This code-only release contains the primary association and matched internal
cross-validation scripts. It does not contain MIMIC-IV data, patient-level
inputs, predictions, or result files. It is not a raw-data cohort extraction
pipeline.

Use Python 3.11 and install the packages in `requirements.txt`. Authorized
MIMIC-IV users must independently derive the analysis inputs under their data
use agreement, using the manuscript's supplementary definitions. Keep these
files local under `rebuild20261026/outputs/`:

- `primary_reconstructed.csv`: primary analytic cohort.
- `all_candidates_reconstructed.csv`: eligibility and missingness comparison.
- `creatinine_observations.csv`: timestamped creatinine measurements.

From the repository root, run:

```sh
python rebuild20261026/analyze.py
python bmcmedicine20261029/matched_cv.py
```

Do not commit inputs or generated outputs. The scripts are not an external
validation of the manuscript findings.
