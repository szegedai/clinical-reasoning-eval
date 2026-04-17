## Response format

Respond with a **single JSON object** and nothing else. No text before or after the JSON.

```json
{
  "assessment": "<string>",
  "delta": "<string>",
  "differential": [
    {"diagnosis": "<string>", "confidence": <float>},
    {"diagnosis": "<string>", "confidence": <float>},
    {"diagnosis": "<string>", "confidence": <float>},
    {"diagnosis": "<string>", "confidence": <float>},
    {"diagnosis": "<string>", "confidence": <float>}
  ],
  "key_findings": [<int>],
  "actions": [{"action": "<string>", "detail": "<string>"}],
  "confident_in_diagnosis": <bool>
}
```

### Field specifications

**assessment** (required): 1-3 sentences summarizing the overall clinical picture.

**delta** (required): 1-2 sentences describing your overall clinical impression and the key reasoning behind your leading diagnosis.

**differential** (required): Exactly 5 diagnoses ranked by likelihood. The first entry is your working diagnosis. You must always provide exactly 5 entries.
- `diagnosis`: standard medical terminology (e.g. "Pulmonary embolism", "Right lower lobe pneumonia")
- `confidence`: 0.0 to 1.0. All five confidences must sum to exactly 1.0.

**key_findings** (required): 3-10 event indices that are most diagnostically significant to your assessment. Reference events by their `[N]` index number.

**actions** (required): 0-3 recommended next clinical actions. If your recommendation is to wait and observe without a specific intervention, leave this array empty.
- `action`: one of the action keys below
- `detail`: short free-text specifying what exactly (e.g. "CT abdomen with contrast", "heparin IV bolus", "surgery")

| Action key | Description |
|---|---|
| `order_labs` | Order laboratory tests |
| `order_imaging` | Order imaging studies |
| `order_microbiology` | Order microbiology cultures/tests |
| `administer_medication` | Administer a medication |
| `start_prescription` | Start a new prescription |
| `perform_procedure` | Perform a procedure |
| `admit_patient` | Admit the patient |
| `discharge_patient` | Discharge the patient |

**confident_in_diagnosis** (required): `true` or `false`. Based on the evidence provided, are you confident enough in your leading diagnosis that you would recommend initiating definitive management?

### Example response

```json
{
  "assessment": "CTPA shows a saddle embolus extending into bilateral pulmonary arteries. Combined with the elevated D-dimer, tachycardia, and pleuritic chest pain, this confirms acute PE.",
  "delta": "The combination of elevated D-dimer, tachycardia, pleuritic chest pain, and CTPA showing a saddle embolus strongly supports acute PE as the primary diagnosis.",
  "differential": [
    {"diagnosis": "Acute pulmonary embolism", "confidence": 0.88},
    {"diagnosis": "Acute coronary syndrome", "confidence": 0.05},
    {"diagnosis": "Pneumothorax", "confidence": 0.03},
    {"diagnosis": "Aortic dissection", "confidence": 0.02},
    {"diagnosis": "Pericarditis", "confidence": 0.02}
  ],
  "key_findings": [3, 11, 19],
  "actions": [
    {"action": "administer_medication", "detail": "heparin IV bolus"},
    {"action": "order_labs", "detail": "troponin, BNP, ABG"}
  ],
  "confident_in_diagnosis": true
}
```