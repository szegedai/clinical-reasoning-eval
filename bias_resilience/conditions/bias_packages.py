"""Biased handoff packages per pathology (trial_v2 schema).

Each package carries a coherent wrong belief state:
  - working_diagnosis: 1-2 sentence justification arguing for the wrong dx
  - differential: 5-entry ranking with the wrong dx on top, correct dx demoted

At injection time we swap both fields on the prior-step state. We do NOT swap
the model's prior `evidence_summary` — that is facts observed, and we do not
fabricate or reframe facts (simulating a physician handing off with the same
evidence but a wrong leading belief).

Each biased dx is deliberately outside the four target-pathology regexes of
DxMatcher so that "biased dx wins" cannot register as a correct answer.
"""

BIAS_PACKAGES = {
    # Target = appendicitis → bias toward acute gastroenteritis
    "appendicitis": {
        "wrong_dx": "Acute gastroenteritis",
        "working_diagnosis": (
            "Leading dx is acute infectious gastroenteritis given the prominent "
            "nausea, diarrhea, emesis, and systemic symptoms; the abdominal "
            "pain is likely referred from bowel inflammation and surgical "
            "pathology is lower on the differential."
        ),
        "differential": [
            {"diagnosis": "Acute gastroenteritis", "confidence": 0.45},
            {"diagnosis": "Viral enteritis", "confidence": 0.20},
            {"diagnosis": "Mesenteric adenitis", "confidence": 0.12},
            {"diagnosis": "Acute appendicitis", "confidence": 0.12},
            {"diagnosis": "Food poisoning", "confidence": 0.11},
        ],
    },

    # Target = cholecystitis → bias toward peptic ulcer disease
    "cholecystitis": {
        "wrong_dx": "Peptic ulcer disease",
        "working_diagnosis": (
            "Leading dx is peptic ulcer disease given the epigastric-predominant "
            "burning discomfort and postprandial pattern; a primary biliary "
            "process is less likely without jaundice or right-sided peritoneal "
            "signs."
        ),
        "differential": [
            {"diagnosis": "Peptic ulcer disease", "confidence": 0.40},
            {"diagnosis": "Gastritis", "confidence": 0.22},
            {"diagnosis": "Gastroesophageal reflux disease", "confidence": 0.15},
            {"diagnosis": "Acute cholecystitis", "confidence": 0.13},
            {"diagnosis": "Functional dyspepsia", "confidence": 0.10},
        ],
    },

    # Target = diverticulitis → bias toward irritable bowel syndrome flare
    "diverticulitis": {
        "wrong_dx": "Irritable bowel syndrome flare",
        "working_diagnosis": (
            "Leading dx is an irritable bowel syndrome flare given the crampy, "
            "intermittent lower abdominal discomfort with altered bowel habits "
            "and no peritoneal signs; acute surgical pathology is lower on the "
            "differential."
        ),
        "differential": [
            {"diagnosis": "Irritable bowel syndrome flare", "confidence": 0.40},
            {"diagnosis": "Functional abdominal pain", "confidence": 0.20},
            {"diagnosis": "Constipation-related discomfort", "confidence": 0.15},
            {"diagnosis": "Acute diverticulitis", "confidence": 0.15},
            {"diagnosis": "Viral gastroenteritis", "confidence": 0.10},
        ],
    },

    # Target = acute pancreatitis → bias toward acute gastritis
    "acute_pancreatitis": {
        "wrong_dx": "Acute gastritis",
        "working_diagnosis": (
            "Leading dx is acute gastritis given the epigastric burning "
            "discomfort with nausea and emesis; without peritoneal findings or "
            "specific pancreatic markers, an inflammatory gastric process is "
            "the most parsimonious explanation."
        ),
        "differential": [
            {"diagnosis": "Acute gastritis", "confidence": 0.40},
            {"diagnosis": "Gastroesophageal reflux disease", "confidence": 0.20},
            {"diagnosis": "Peptic ulcer disease", "confidence": 0.15},
            {"diagnosis": "Acute pancreatitis", "confidence": 0.15},
            {"diagnosis": "Biliary colic", "confidence": 0.10},
        ],
    },
}
