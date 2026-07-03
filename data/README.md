# Polish companies bankruptcy data setup

The ARFF files are **not** included in this repository.

## Expected layout

After setup, this directory should contain:

```text
data/
├── 4year.arff   # training cohort (required)
└── 5year.arff   # test cohort (required)
```

Other year files (`1year.arff`–`3year.arff`) are optional and not used by the pipeline.

## Download

1. Open the [UCI Polish companies bankruptcy dataset](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data).
2. Download the data files and extract `4year.arff` and `5year.arff` into this `data/` folder.

## Verify

From the repo root:

```bash
python -c "from pathlib import Path; p=Path('data'); ok=all((p/f).is_file() for f in ['4year.arff','5year.arff']); print('OK' if ok else 'Missing data')"
```

Then run:

```bash
python bankruptcy_classifier.py
```
