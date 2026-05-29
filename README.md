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
  pathologies.yaml        #   ICD codes, HPI patterns, dx_match/dx_gracious regexes for all pathologies
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
  dx_matcher.py           #   regex-based diagnosis matching (dx_match + dx_gracious tiers)
  freetextdx.py           #   parse free-text discharge diagnosis from note sections

run_replay.py             # CLI: run replay on a batch of patients
utils/check_bq_usage.py   # BigQuery cost monitoring

bias_resilience/          # Bias-resilience experiment framework
  config.py               #   model configs, condition registry, paths
  cli.py                  #   CLI entrypoint (python -m bias_resilience.cli)
  runner.py               #   per-patient orchestration, output JSON
  replay.py               #   core temporal replay loop with injection hooks
  schema.py               #   LLM response parsing and validation
  anchors.py              #   anchor resolution (post_pe, post_first_labs, post_imaging)
  demote.py               #   runner-up wrong-dx extraction from baseline
  dx_dedup.py             #   compound diagnosis deduplication
  cohort.py               #   cohort manifest loading and filtering
  conditions/             #   bias injection conditions (baseline, struc_belief, struc_consult, pushback)
  prompts/                #   step prompt templates
  tests/                  #   unit tests (no LLM calls)
```

## Usage

### 1. Extract timelines (requires BQ access)

```bash
# Recommended: require ED admission, discharge note, no HPI leak, dx confirmed in discharge note
python dataset/create_cohort.py --pathology appendicitis --exclude-hpi-dx --require-freetextdx --limit 100 --output cohort.txt

# Check BQ cost before running
python dataset/create_cohort.py --pathology appendicitis --require-freetextdx --output /dev/null --dry-run

# See dataset/README.md for all flags and how they stack
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

### 4. Bias-resilience experiments

```bash
# Baseline run on a cohort
python -m bias_resilience.cli \
  --run-id baseline_appendicitis \
  --model gemini-2.5-flash \
  --condition baseline \
  --pathology appendicitis \
  --cohort-file path/to/cohort.txt \
  --max-hours 48

# Structured belief injection
python -m bias_resilience.cli \
  --run-id struc_belief_appendicitis \
  --model gemini-2.5-flash \
  --condition struc_belief \
  --pathology appendicitis \
  --cohort-file path/to/cohort.txt \
  --anchor post_imaging \
  --max-hours 48
```

Set `LOCAL_LLM_BASE_URL` and `LOCAL_LLM_MODEL_ID` in your `.env` to use a local model (Ollama, vLLM, etc.) with `--model local`.

## How the replay works

Each patient timeline is split into clinical steps (arrival → triage → exam → labs/imaging → ...). At each step, new events are presented to the LLM as a user message in a multi-turn conversation. The LLM responds with a JSON containing its current assessment, differential diagnosis with confidence scores, key findings, and recommended actions.

The `openai` SDK with `base_url` is used for all providers (OpenAI, Gemini, local models via vLLM/Ollama).

---

Parts of this codebase were developed with the assistance of an AI coding tool. Disclosed in accordance with ACL policies.
