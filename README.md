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
dataset/                  # BigQuery → timeline extraction
  create_cohort.py        #   query hadm_ids by ICD code + filters
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

run_replay.py             # CLI: run replay on a batch of patients
collect_results.py        # Collect per-step results into Excel
utils/check_bq_usage.py   # BigQuery cost monitoring
```

## Usage

### 1. Extract timelines (requires BQ access)

```bash
# Create a cohort of hadm_ids
python dataset/create_cohort.py --icd-prefix K35 K37 --limit 100 --output cohort.txt

# Extract timelines in batch
python dataset/timeline_batch.py --file cohort.txt --output-dir timelines/
```

### 2. Run replay against an LLM

Create a config file (see `replay_config.yaml` for defaults):

```yaml
model: gemini-2.0-flash
base_url: https://generativelanguage.googleapis.com/v1beta/openai/
api_key_env: GEMINI_API_KEY    # reads API key from this env var
temperature: 0.0

prompts:
  system: system_prompt.md
  step: step_prompt.md

chunker:
  stop_at:
    event_type: SERVICE
    description: "Service: SURG"
```

Run:

```bash
export GEMINI_API_KEY=...
python run_replay.py -c replay_config.yaml --timeline-dir timelines/ -o results/run1/
```

Resume an interrupted run:

```bash
python run_replay.py -c replay_config.yaml --timeline-dir timelines/ -o results/run1/ --skip-existing
```

### 3. Collect results

```bash
python collect_results.py results/run1/
# → results/run1/results.xlsx
```

## How the replay works

Each patient timeline is split into clinical steps (arrival → triage → exam → labs/imaging → ...). At each step, new events are presented to the LLM as a user message in a multi-turn conversation. The LLM responds with a JSON containing its current assessment, differential diagnosis with confidence scores, key findings, and recommended actions.

The `openai` SDK with `base_url` is used for all providers (OpenAI, Gemini, local models via vLLM/Ollama).
