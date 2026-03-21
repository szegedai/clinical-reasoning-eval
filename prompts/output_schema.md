## Response Format

Respond with a **single JSON object** and nothing else. No text before or after the JSON.

```json
{
  "assessment": "<string>",
  "differential": [
    {"diagnosis": "<string>", "confidence": <float>}
  ],
  "key_findings": [<int>],
  "recommended_actions": [
    {"action": "<string>", "detail": "<string>"}
  ]
}
```

### Field specifications

**assessment** (required): 2-4 sentences of clinical reasoning. What is the clinical picture so far? What changed with the new information?

**differential** (required): 1-5 diagnoses ranked by likelihood. The first entry is your working diagnosis.
- `diagnosis`: Standard medical terminology (e.g. "Acute appendicitis", "Right lower lobe pneumonia")
- `confidence`: 0.0 to 1.0. All confidences should sum to ≤ 1.0.

**key_findings** (required): List of event indices (e.g. `[0, 3, 12]`) identifying the most diagnostically relevant findings from ALL events seen so far. Reference events by their `[N]` index number. Include findings from any step, not just the most recent one.

**recommended_actions** (required): 1-3 recommended next clinical actions. Each action has:
- `action`: One of the values below.
- `detail`: Brief free-text description of the specific action (e.g. "CT abdomen/pelvis with IV contrast").

### Action menu

| Action key | Description |
|---|---|
| `order_labs` | Order laboratory tests |
| `order_imaging` | Order imaging studies |
| `order_microbiology` | Order microbiology cultures/tests |
| `administer_medication` | Administer a medication |
| `start_prescription` | Start a new prescription |
| `consult_service` | Request a specialty consultation |
| `monitor_vitals` | Monitor vital signs |
| `perform_procedure` | Perform a procedure |
| `admit_patient` | Admit the patient |
| `discharge_patient` | Discharge the patient |

### Example response

```json
{
  "assessment": "The clinical picture is consistent with acute appendicitis. The patient presents with right lower quadrant pain, low-grade fever, and leukocytosis. The CT findings of a dilated appendix with periappendiceal fat stranding confirm the diagnosis.",
  "differential": [
    {"diagnosis": "Acute appendicitis", "confidence": 0.85},
    {"diagnosis": "Mesenteric lymphadenitis", "confidence": 0.08},
    {"diagnosis": "Right ovarian pathology", "confidence": 0.04}
  ],
  "key_findings": [2, 14, 23],
  "recommended_actions": [
    {"action": "consult_service", "detail": "General surgery for appendectomy evaluation"},
    {"action": "administer_medication", "detail": "IV piperacillin-tazobactam for empiric coverage"}
  ]
}
```