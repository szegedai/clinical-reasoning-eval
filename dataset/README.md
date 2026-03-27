# dataset/

BigQuery → cohort → timeline extraction pipeline.

## Scripts

| Script | What it does |
|--------|-------------|
| `create_cohort.py` | Query BQ for hadm_ids matching a pathology; write to a text file |
| `timeline.py` | Extract a single patient's timeline from BQ (used by batch) |
| `timeline_batch.py` | Batch-extract timelines for a list of hadm_ids |

---

## create_cohort.py

### Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--pathology NAME` | — | Load ICD codes + HPI/dx patterns from `configs/pathologies.yaml` |
| `--icd-range K35,K37` | — | Explicit ICD prefixes (mutually exclusive with `--pathology`) |
| `--output FILE` | — | Write one hadm_id per line |
| `--no-require-note` | require | Skip the discharge note JOIN (more results, no note filter) |
| `--no-require-ed` | require | Skip the ED stay JOIN (includes non-ED admissions) |
| `--limit N` | none | Cap results at N hadm_ids |
| `--exclude-hpi-dx` | off | Drop patients whose HPI text mentions the diagnosis name |
| `--hpi-pattern REGEX` | from yaml | BQ re2 pattern for `--exclude-hpi-dx` (auto-loaded with `--pathology`) |
| `--require-freetextdx` | off | Keep only patients whose discharge note free-text diagnosis matches pathology regexes |
| `--project ID` | gcloud default | GCP billing project |
| `--dry-run` | off | Print SQL + estimated scan cost, don't execute |

### How filters stack

Each flag adds a JOIN or WHERE clause to the BQ query:

```
ICD seq=1 match          (always)
  + discharge note JOIN  (--pathology / --no-require-note off)
  + ED stay JOIN         (--no-require-ed off)
  + HPI regex filter     (--exclude-hpi-dx)
  + freetextdx filter    (--require-freetextdx)
```

`--exclude-hpi-dx` removes patients where the HPI already names the diagnosis — useful for leak-free evaluation (the HPI is shown to the LLM at step 1).

`--require-freetextdx` keeps only patients where the free-text primary diagnosis in the discharge note matches the pathology's `dx_match` or `dx_gracious` patterns. This removes miscoded patients (e.g. TGA patients under TIA, biliary disease under pancreatitis).

### Recommended config

For a clean evaluation cohort:

```bash
python dataset/create_cohort.py \
  --pathology appendicitis \
  --exclude-hpi-dx \
  --require-freetextdx \
  --limit 100 \
  --output cohorts/appendicitis_ids.txt
```

For a quick exploratory cohort (no text filtering, cheaper query):

```bash
python dataset/create_cohort.py \
  --pathology appendicitis \
  --limit 200 \
  --output cohorts/appendicitis_ids.txt
```

Check cost before running:

```bash
python dataset/create_cohort.py --pathology appendicitis --require-freetextdx --output /dev/null --dry-run
```

### BQ cost

`--exclude-hpi-dx` and `--require-freetextdx` both scan `mimiciv_note.discharge` (~2–4 GB). The main cost driver is `chartevents` in `timeline_batch.py` (~43.5 GB per run regardless of patient count).

---

## timeline_batch.py

```bash
python dataset/timeline_batch.py \
  --file cohorts/appendicitis_ids.txt \
  --output-dir timelines/appendicitis/
```

| Flag | Default | Effect |
|------|---------|--------|
| `--file FILE` | — | Text file with one hadm_id per line |
| `--output-dir DIR` | — | Write `timeline_<hadm_id>.csv` per patient |
| `--batch-size N` | 50 | hadm_ids per BQ query (all batches cost the same) |
| `--project ID` | gcloud default | GCP billing project |

One batch query scans ~43.5 GB regardless of patient count — batching all patients together is always cheaper than running individually.
