# clinical-reasoning-eval

Evaluating LLMs through temporal replay of diagnostic scenarios using MIMIC-IV data.

Builds chronological patient timelines from MIMIC-IV, replays them to LLMs step by step, and collects diagnostic reasoning at each step.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Google Cloud credentials for BigQuery access (dataset extraction only).

## Repo structure

```
configs/                  # Configuration
  pathologies.yaml        #   ICD codes, HPI patterns, dx_aliases for all pathologies
  replay_config*.yaml     #   replay runner configs (model, chunker params, etc.)

dataset/                  # BigQuery → timeline extraction
  create_cohort.py        #   query hadm_ids by ICD code or pathology config + filters
  timeline.py             #   single-patient timeline from BQ
  timeline_batch.py       #   batch extraction (cheaper, same BQ scan cost)

temporal_replay/          # Timeline → chunked prompts
  chunker.py              #   splits timeline CSV into clinical steps
  formatter.py            #   formats events into human-readable text
  renderer.py             #   fills prompt templates with event data

prompts/                  # Prompt templates
  system_prompt.md        #   system message (role, task, output format)
  step_prompt.md          #   per-step user message template
  output_schema.md        #   expected JSON response schema

llm/                      # LLM calling
  runner.py               #   PatientRunner: replay loop, multi-turn conversation
  parser.py               #   JSON extraction + field validation

analysis/                 # Result analysis and visualization
  analyze_results.py      #   accuracy/confidence plots from replay JSONs
  collect_results.py      #   collect per-step results into Excel

run_replay.py             # CLI: run replay on a batch of patients
utils/check_bq_usage.py   # BigQuery cost monitoring
```

## Usage

### 1. Extract timelines (requires BQ access)

```bash
# Create a cohort using pathology config (loads ICD codes from YAML)
python dataset/create_cohort.py --pathology appendicitis --limit 100 --output cohort.txt

# Or with explicit ICD prefixes
python dataset/create_cohort.py --icd-range K35,K37 --limit 100 --output cohort.txt

# Exclude patients whose HPI mentions the diagnosis
python dataset/create_cohort.py --pathology appendicitis --exclude-hpi-dx --output cohort_clean.txt

# Extract timelines in batch
python dataset/timeline_batch.py --file cohort.txt --output-dir timelines/
```

### 2. Run replay against an LLM

See `configs/replay_config_gemini.yaml` for a full example. Key options:

```yaml
model: gemini-2.0-flash
base_url: https://generativelanguage.googleapis.com/v1beta/openai/
api_key_env: GEMINI_API_KEY
temperature: 0.0
max_steps: 20

chunker:
  max_events: 25
  max_event_types: 3
  max_hours: 4.0
  stop_at:
    event_type: SERVICE
    description: "Service: SURG"
  exclude_sources: [ICU]
  exclude_event_types: [DISCHARGE_DX, DISCHARGE_FREETEXTDX]
  max_chunks: 50
```

Run:

```bash
export GEMINI_API_KEY=...
python run_replay.py -c configs/replay_config_gemini.yaml --timeline-dir timelines/ -o results/run1/

# Resume an interrupted run
python run_replay.py -c configs/replay_config_gemini.yaml --timeline-dir timelines/ -o results/run1/ --skip-existing
```

### 3. Analyze results

```bash
# Accuracy plots + Excel for one pathology
python analysis/analyze_results.py results/run1/ --pathology appendicitis -o results/

# Multiple pathologies (multi-panel plots)
python analysis/analyze_results.py results/appendicitis_gemini results/cholecystitis_gemini \
  --pathology appendicitis cholecystitis -o results/

# Collect per-step results into Excel
python analysis/collect_results.py results/run1/
```

## How the replay works

Each patient timeline is split into clinical steps (arrival → triage → exam → labs/imaging → ...). At each step, new events are presented to the LLM as a user message in a multi-turn conversation. The LLM responds with a JSON containing its current assessment, differential diagnosis with confidence scores, key findings, and recommended actions.

The `openai` SDK with `base_url` is used for all providers (OpenAI, Gemini, local models via vLLM/Ollama).
