"""
Builds a chronological event timeline for a single hospital admission (hadm_id)
by pulling from all four MIMIC-IV modules: ED, HOSP, ICU, NOTE.

Output columns:
  subject_id    – patient identifier
  hadm_id       – hospital admission identifier
  event_time    – absolute timestamp of the event
  elapsed_hours – hours since the first recorded event (t0)
  source        – module: ED | HOSP | ICU | NOTE | LAB
  event_type    – structured category (e.g. ADMISSION, LAB_RESULT, ICU_VITAL)
  description   – human-readable event summary
  value         – numeric value (if applicable)
  unit          – unit string (if applicable)
  flag          – e.g. "abnormal" for labs, "1" for ICU warning

Usage:
  python timeline.py --hadm_id 20973395
  python timeline.py --hadm_id 20973395 --project my-gcp-project
"""

import argparse
import re
import sys
from typing import Optional

from google.cloud import bigquery
import pandas as pd


HOSP_DS = "physionet-data.mimiciv_3_1_hosp"
ICU_DS  = "physionet-data.mimiciv_3_1_icu"
ED_DS   = "physionet-data.mimiciv_ed"
NOTE_DS = "physionet-data.mimiciv_note"

VITAL_ITEM_IDS = [
    220045,  # Heart Rate
    220210,  # Respiratory Rate
    224690,  # Respiratory Rate Total
    220277,  # SpO2 pulse oximetry
    223761,  # Temperature Fahrenheit
    223762,  # Temperature Celsius
    220050,  # Arterial BP Systolic
    220051,  # Arterial BP Diastolic
    220052,  # Arterial BP Mean
    220179,  # Non-invasive BP Systolic
    220180,  # Non-invasive BP Diastolic
    220181,  # Non-invasive BP Mean
    226730,  # GCS Total
    220739,  # GCS – Eye Opening
    223900,  # GCS – Verbal Response
    223901,  # GCS – Motor Response
]

VITAL_IDS_STR = ", ".join(str(i) for i in VITAL_ITEM_IDS)

ANCHORED_EVENT_ORDER = [
    "ED_ARRIVAL",
    "DISCHARGE_HPI",
    "TRIAGE_COMPLAINT",
    "TRIAGE_PAIN",
    "ED_VITALS",
    "TRIAGE_VITAL",
    "TRIAGE_ACUITY",
    "DISCHARGE_PE",
]

_ANCHORED_ORDER_MAP = {et: i for i, et in enumerate(ANCHORED_EVENT_ORDER)}
_ANCHORED_DEFAULT = len(ANCHORED_EVENT_ORDER)


def sort_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a timeline DataFrame with custom ordering for tied anchored events."""
    sort_key = df["event_type"].map(_ANCHORED_ORDER_MAP).fillna(_ANCHORED_DEFAULT)
    df = df.assign(_sort_key=sort_key)
    df.sort_values(
        ["event_time", "_sort_key", "source", "event_type"],
        inplace=True,
    )
    df.drop(columns="_sort_key", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_query() -> str:

    notes_sql = f"""
  UNION ALL
  -- NOTE: Imaging study performed (charttime = when the study was done)
  SELECT
    nr.hadm_id,
    nr.charttime AS event_time,
    'NOTE' AS source,
    'IMAGING_STUDY' AS event_type,
    CONCAT(nr.note_type, ' seq=', CAST(nr.note_seq AS STRING),
           ' | note_id=', nr.note_id) AS description,
    CAST(NULL AS STRING) AS value,
    CAST(NULL AS STRING) AS unit,
    CAST(NULL AS STRING) AS flag,
    'exact' AS time_precision
  FROM `{NOTE_DS}.radiology` nr
  WHERE nr.hadm_id = @hadm_id

  UNION ALL
  -- NOTE: Radiology report available (storetime = when the report was signed)
  SELECT
    nr.hadm_id,
    COALESCE(nr.storetime, nr.charttime) AS event_time,
    'NOTE' AS source,
    'RADIOLOGY_REPORT' AS event_type,
    CONCAT(nr.note_type, ' seq=', CAST(nr.note_seq AS STRING), ': ', nr.text,
           ' | note_id=', nr.note_id) AS description,
    CAST(NULL AS STRING) AS value,
    CAST(NULL AS STRING) AS unit,
    CAST(NULL AS STRING) AS flag,
    CASE WHEN nr.storetime IS NOT NULL THEN 'exact' ELSE 'anchored' END AS time_precision
  FROM `{NOTE_DS}.radiology` nr
  WHERE nr.hadm_id = @hadm_id

  UNION ALL
  -- NOTE: Discharge summaries
  SELECT
    nd.hadm_id,
    nd.charttime, 'NOTE', 'DISCHARGE_NOTE',
    CONCAT(nd.note_type, ' seq=', CAST(nd.note_seq AS STRING), ': ', nd.text),
    NULL, NULL, NULL, 'date_only'
  FROM `{NOTE_DS}.discharge` nd
  WHERE nd.hadm_id = @hadm_id
"""

    rx_sql = f"""
  UNION ALL
  -- HOSP: Prescription orders (when the order was started)
  SELECT
    p.hadm_id,
    p.starttime, 'HOSP', 'RX_START',
    CONCAT('Rx: ', p.drug,
           ' | ', COALESCE(CAST(p.dose_val_rx AS STRING), '?'),
           ' ', COALESCE(p.dose_unit_rx, ''),
           ' | Route: ', COALESCE(p.route, '?')),
    NULL, NULL, NULL, 'exact'
  FROM `{HOSP_DS}.prescriptions` p
  WHERE p.hadm_id = @hadm_id
    AND p.starttime IS NOT NULL
"""

    medrecon_sql = f"""
  UNION ALL
  -- ED: Home medication reconciliation (what the patient was taking at home)
  SELECT
    es.hadm_id,
    m.charttime, 'ED', 'MED_RECON',
    CONCAT('Home med: ', m.name,
           COALESCE(CONCAT(' (', m.etcdescription, ')'), '')),
    NULL, NULL, NULL, 'exact'
  FROM `{ED_DS}.medrecon` m
  JOIN ed_stay es ON m.stay_id = es.stay_id
"""

    return f"""
-- ── Anchor CTEs ───────────────────────────────────────────────────────────────
WITH
adm AS (
  SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    admission_type,
    admission_location,
    discharge_location
  FROM `{HOSP_DS}.admissions`
  WHERE hadm_id = @hadm_id
),

-- ED stay may not exist for all admissions (e.g. direct admits)
ed_stay AS (
  SELECT hadm_id, stay_id, intime, outtime, disposition
  FROM `{ED_DS}.edstays`
  WHERE hadm_id = @hadm_id
),

-- t0 = earliest timestamp: ED arrival if it exists, otherwise admittime
t0_cte AS (
  SELECT
    adm.hadm_id,
    LEAST(
      COALESCE((SELECT MIN(intime) FROM ed_stay WHERE ed_stay.hadm_id = adm.hadm_id), adm.admittime),
      adm.admittime
    ) AS t0
  FROM adm
),

-- ── Event stream (UNION ALL of all sources) ───────────────────────────────────
all_events AS (

  -- ── ED MODULE ─────────────────────────────────────────────────────────────

  -- ED: Arrival
  SELECT
    hadm_id,
    intime AS event_time,
    'ED' AS source,
    'ED_ARRIVAL' AS event_type,
    CONCAT('ED arrival | Disposition: ', COALESCE(disposition, 'unknown')) AS description,
    CAST(NULL AS STRING) AS value,
    CAST(NULL AS STRING) AS unit,
    CAST(NULL AS STRING) AS flag,
    'exact' AS time_precision
  FROM ed_stay

  UNION ALL

  -- ED: Departure
  SELECT
    hadm_id,
    outtime, 'ED', 'ED_DEPARTURE',
    'Patient left ED', NULL, NULL, NULL, 'exact'
  FROM ed_stay
  WHERE outtime IS NOT NULL

  UNION ALL

  -- ED: Triage

  -- Chief complaint
  SELECT
    es.hadm_id,
    es.intime, 'ED', 'TRIAGE_COMPLAINT',
    COALESCE(t.chiefcomplaint, 'unknown'),
    CAST(NULL AS STRING), CAST(NULL AS STRING), CAST(NULL AS STRING), 'anchored'
  FROM `{ED_DS}.triage` t
  JOIN ed_stay es ON t.stay_id = es.stay_id

  UNION ALL

  -- Acuity (ESI level 1–5)
  SELECT
    es.hadm_id,
    es.intime, 'ED', 'TRIAGE_ACUITY',
    CONCAT('ESI acuity level: ', COALESCE(CAST(t.acuity AS STRING), 'unknown')),
    CAST(t.acuity AS STRING), CAST(NULL AS STRING), CAST(NULL AS STRING), 'anchored'
  FROM `{ED_DS}.triage` t
  JOIN ed_stay es ON t.stay_id = es.stay_id
  WHERE t.acuity IS NOT NULL

  UNION ALL

  -- Pain score (0–10, self-reported)
  SELECT
    es.hadm_id,
    es.intime, 'ED', 'TRIAGE_PAIN',
    CONCAT('Pain: ', t.pain, '/10'),
    t.pain, CAST(NULL AS STRING), CAST(NULL AS STRING), 'anchored'
  FROM `{ED_DS}.triage` t
  JOIN ed_stay es ON t.stay_id = es.stay_id
  WHERE t.pain IS NOT NULL

  UNION ALL

  -- Temperature (°F)
  SELECT
    es.hadm_id,
    es.intime, 'ED', 'TRIAGE_VITAL',
    CONCAT('Temperature: ', CAST(t.temperature AS STRING), ' °F'),
    CAST(t.temperature AS STRING), '°F', CAST(NULL AS STRING), 'anchored'
  FROM `{ED_DS}.triage` t
  JOIN ed_stay es ON t.stay_id = es.stay_id
  WHERE t.temperature IS NOT NULL

  UNION ALL

  -- Heart rate (bpm)
  SELECT
    es.hadm_id,
    es.intime, 'ED', 'TRIAGE_VITAL',
    CONCAT('Heart Rate: ', CAST(t.heartrate AS STRING), ' bpm'),
    CAST(t.heartrate AS STRING), 'bpm', CAST(NULL AS STRING), 'anchored'
  FROM `{ED_DS}.triage` t
  JOIN ed_stay es ON t.stay_id = es.stay_id
  WHERE t.heartrate IS NOT NULL

  UNION ALL

  -- Respiratory rate (breaths/min)
  SELECT
    es.hadm_id,
    es.intime, 'ED', 'TRIAGE_VITAL',
    CONCAT('Respiratory Rate: ', CAST(t.resprate AS STRING), ' brpm'),
    CAST(t.resprate AS STRING), 'brpm', CAST(NULL AS STRING), 'anchored'
  FROM `{ED_DS}.triage` t
  JOIN ed_stay es ON t.stay_id = es.stay_id
  WHERE t.resprate IS NOT NULL

  UNION ALL

  -- SpO2 (%)
  SELECT
    es.hadm_id,
    es.intime, 'ED', 'TRIAGE_VITAL',
    CONCAT('SpO2: ', CAST(t.o2sat AS STRING), ' %'),
    CAST(t.o2sat AS STRING), '%', CAST(NULL AS STRING), 'anchored'
  FROM `{ED_DS}.triage` t
  JOIN ed_stay es ON t.stay_id = es.stay_id
  WHERE t.o2sat IS NOT NULL

  UNION ALL

  -- Blood pressure (systolic / diastolic, mmHg)
  SELECT
    es.hadm_id,
    es.intime, 'ED', 'TRIAGE_VITAL',
    CONCAT('BP: ', CAST(t.sbp AS STRING), '/', CAST(t.dbp AS STRING), ' mmHg'),
    CAST(t.sbp AS STRING), 'mmHg', CAST(NULL AS STRING), 'anchored'
  FROM `{ED_DS}.triage` t
  JOIN ed_stay es ON t.stay_id = es.stay_id
  WHERE t.sbp IS NOT NULL AND t.dbp IS NOT NULL

  UNION ALL

  -- ED: Serial vital signs
  SELECT
    es.hadm_id,
    v.charttime, 'ED', 'ED_VITALS',
    CONCAT(
      'HR: ',    COALESCE(CAST(v.heartrate   AS STRING), '?'),
      ' | RR: ', COALESCE(CAST(v.resprate    AS STRING), '?'),
      ' | SpO2: ',COALESCE(CAST(v.o2sat      AS STRING), '?'), '%',
      ' | BP: ', COALESCE(CAST(v.sbp         AS STRING), '?'),
      '/',       COALESCE(CAST(v.dbp         AS STRING), '?'),
      ' | Temp: ',COALESCE(CAST(v.temperature AS STRING), '?'), 'F',
      ' | Pain: ',COALESCE(CAST(v.pain       AS STRING), '?')
    ),
    NULL, NULL, NULL, 'exact'
  FROM `{ED_DS}.vitalsign` v
  JOIN ed_stay es ON v.stay_id = es.stay_id

  UNION ALL

  -- ED: Diagnoses — the diagnosis table no timestamp column at all. Anchored to ed outtime.
  SELECT
    es.hadm_id,
    es.outtime, 'ED', 'ED_DIAGNOSIS',
    CONCAT('ED Dx [ICD', CAST(d.icd_version AS STRING), ']: ',
           d.icd_code, ' – ', d.icd_title),
    NULL, NULL, NULL, 'anchored'
  FROM `{ED_DS}.diagnosis` d
  JOIN ed_stay es ON d.stay_id = es.stay_id

  UNION ALL

  -- ED: Pyxis dispensing (ED automated medication dispense)
  SELECT
    es.hadm_id,
    py.charttime, 'ED', 'ED_PYXIS',
    CONCAT('ED dispense: ', py.name,
           COALESCE(CONCAT(' (gsn=', CAST(py.gsn AS STRING), ')'), '')),
    NULL, NULL, NULL, 'exact'
  FROM `{ED_DS}.pyxis` py
  JOIN ed_stay es ON py.stay_id = es.stay_id

  {medrecon_sql}

  -- ── HOSP MODULE ───────────────────────────────────────────────────────────

  UNION ALL

  -- HOSP: Hospital admission event
  SELECT
    adm.hadm_id,
    adm.admittime, 'HOSP', 'ADMISSION',
    CONCAT('Hospital admission | Type: ', adm.admission_type,
           ' | From: ', COALESCE(adm.admission_location, 'unknown')),
    NULL, NULL, NULL, 'exact'
  FROM adm

  UNION ALL

  -- HOSP: Physical location transfers (ED → ICU → floor → discharge)
  SELECT
    tr.hadm_id,
    tr.intime, 'HOSP', 'TRANSFER',
    CONCAT(UPPER(tr.eventtype), ': ', COALESCE(tr.careunit, 'unknown')),
    NULL, NULL, NULL, 'exact'
  FROM `{HOSP_DS}.transfers` tr
  WHERE tr.hadm_id = @hadm_id

  UNION ALL

  -- HOSP: Clinical service assignments
  SELECT
    sv.hadm_id,
    sv.transfertime, 'HOSP', 'SERVICE',
    CONCAT('Service: ', sv.curr_service,
           COALESCE(CONCAT(' (from ', sv.prev_service, ')'), '')),
    NULL, NULL, NULL, 'exact'
  FROM `{HOSP_DS}.services` sv
  WHERE sv.hadm_id = @hadm_id

  UNION ALL

  -- HOSP: Lab events are split into two distinct event types:
  --
  --   SPECIMEN_COLLECTED (at charttime) — one row per unique specimen_id.
  --     specimen_id groups all tests drawn from the same physical sample.
  --     charttime = when the specimen was acquired (blood draw / clinical action).
  --
  --   LAB_RESULT (at storetime) — one row per individual test result.
  --     storetime = when the result became available to care providers.
  --     Falls back to charttime when storetime is NULL, marked as 'anchored'.

  -- SPECIMEN_COLLECTED: one event per blood draw
  SELECT
    l.hadm_id,
    MIN(l.charttime), 'LAB', 'SPECIMEN_COLLECTED',
    CONCAT(
      CAST(COUNT(*) AS STRING), ' test(s) drawn (',
      STRING_AGG(dl.label, ', '),
      ') | specimen_id=', CAST(l.specimen_id AS STRING)
    ),
    NULL, NULL, NULL, 'exact'
  FROM `{HOSP_DS}.labevents` l
  JOIN `{HOSP_DS}.d_labitems` dl ON l.itemid = dl.itemid
  WHERE l.hadm_id = @hadm_id
  GROUP BY l.hadm_id, l.specimen_id

  UNION ALL

  -- LAB_RESULT: one event per result, timestamped when available to clinicians
  SELECT
    l.hadm_id,
    COALESCE(l.storetime, l.charttime), 'LAB', 'LAB_RESULT',
    CONCAT(
      dl.label, ': ',
      COALESCE(l.value, 'no result'),
      CASE WHEN l.value IS NOT NULL OR l.valuenum IS NOT NULL
           THEN COALESCE(CONCAT(' ', l.valueuom), '')
           ELSE '' END,
      CASE WHEN l.flag IS NOT NULL
           THEN CONCAT(' [', UPPER(l.flag), ']')
           ELSE '' END,
      ' | specimen_id=', CAST(l.specimen_id AS STRING)
    ),
    CAST(l.valuenum AS STRING),
    l.valueuom,
    l.flag,
    CASE WHEN l.storetime IS NOT NULL THEN 'exact' ELSE 'anchored' END
  FROM `{HOSP_DS}.labevents` l
  JOIN `{HOSP_DS}.d_labitems` dl ON l.itemid = dl.itemid
  WHERE l.hadm_id = @hadm_id

  UNION ALL

  -- HOSP: Microbiology — split into sample collection vs result availability
  --   (mirrors the SPECIMEN_COLLECTED / LAB_RESULT split for lab events)
  --
  --   MICRO_SAMPLE (at charttime/chartdate) — one row per specimen × test.
  --     charttime = when the specimen was collected.
  --
  --   MICRO_RESULT (at storetime/storedate) — one row per specimen × test × organism,
  --     including antibiotic sensitivities. storetime = when the result was finalized.
  --     Falls back to charttime/chartdate when storetime is NULL.

  -- MICRO_SAMPLE: one event per specimen × test
  SELECT
    mb.hadm_id,
    MIN(COALESCE(mb.charttime, mb.chartdate)), 'HOSP', 'MICRO_SAMPLE',
    CONCAT(
      mb.spec_type_desc, ' → ', mb.test_name,
      ' | micro_specimen_id=', CAST(mb.micro_specimen_id AS STRING)
    ),
    NULL, NULL, NULL,
    CASE WHEN MIN(mb.charttime) IS NOT NULL THEN 'exact' ELSE 'anchored' END
  FROM `{HOSP_DS}.microbiologyevents` mb
  WHERE mb.hadm_id = @hadm_id
  GROUP BY mb.hadm_id, mb.micro_specimen_id, mb.spec_type_desc, mb.test_name

  UNION ALL

  -- MICRO_RESULT: one event per specimen × test × organism, with sensitivities
  SELECT
    mb.hadm_id,
    MIN(COALESCE(mb.storetime, mb.storedate, mb.charttime, mb.chartdate)),
    'HOSP', 'MICRO_RESULT',
    CONCAT(
      mb.spec_type_desc, ' → ', mb.test_name,
      CASE WHEN mb.org_name IS NOT NULL
           THEN CONCAT(': ', mb.org_name)
           ELSE ': No growth' END,
      CASE WHEN mb.org_name IS NOT NULL
           THEN COALESCE(
             CONCAT(' | Sensitivities: ',
               STRING_AGG(
                 CONCAT(mb.ab_name, '=', mb.interpretation),
                 ', '
               )
             ), '')
           ELSE '' END,
      COALESCE(CONCAT(' | ', ANY_VALUE(mb.comments)), ''),
      ' | micro_specimen_id=', CAST(mb.micro_specimen_id AS STRING)
    ),
    NULL, NULL, NULL,
    CASE WHEN MIN(mb.storetime) IS NOT NULL THEN 'exact' ELSE 'anchored' END
  FROM `{HOSP_DS}.microbiologyevents` mb
  WHERE mb.hadm_id = @hadm_id
  GROUP BY mb.hadm_id, mb.micro_specimen_id, mb.spec_type_desc, mb.test_name, mb.org_name

  UNION ALL

  -- HOSP: eMAR – actual medication administrations (what was actually given)
  --   em.medication is NULL for IV fluids / IV therapy orders; fall back to
  --   the POE order description in that case.
  SELECT
    em.hadm_id,
    em.charttime, 'HOSP', 'MED_ADMIN',
    CONCAT(
      COALESCE(em.medication,
               CONCAT(p.order_type, ': ', p.order_subtype),
               'Unknown medication'),
      ': ', COALESCE(em.event_txt, 'Administered')
    ),
    NULL, NULL, NULL, 'exact'
  FROM `{HOSP_DS}.emar` em
  LEFT JOIN `{HOSP_DS}.poe` p ON em.poe_id = p.poe_id
  WHERE em.hadm_id = @hadm_id
    AND em.charttime IS NOT NULL

  {rx_sql}

  UNION ALL

  -- HOSP: Discharge diagnoses — diagnoses_icd has NO timestamp column.
  --   Per MIMIC docs: "Diagnoses are billed on hospital discharge."
  --   Anchored to dischtime as the closest real event.
  SELECT
    dx.hadm_id,
    adm.dischtime, 'HOSP', 'DISCHARGE_DX',
    CONCAT(
      'Dx [ICD', CAST(dx.icd_version AS STRING),
      ' #', CAST(dx.seq_num AS STRING), ']: ',
      dx.icd_code, ' – ', dicd.long_title
    ),
    NULL, NULL, NULL, 'anchored'
  FROM `{HOSP_DS}.diagnoses_icd` dx
  JOIN `{HOSP_DS}.d_icd_diagnoses` dicd
    ON dx.icd_code = dicd.icd_code AND dx.icd_version = dicd.icd_version
  JOIN adm ON dx.hadm_id = adm.hadm_id
  WHERE dx.hadm_id = @hadm_id

  UNION ALL

  -- HOSP: ICD-coded procedures — procedures_icd has chartdate (DATE only, no time).
  --   elapsed_hours is left NULL (time_precision='date_only').
  --   event_time is set to midnight of chartdate for sort ordering only.
  --   The actual procedure timestamps come from ICU_PROCEDURE events.
  SELECT
    pr.hadm_id,
    DATETIME(pr.chartdate), 'HOSP', 'PROCEDURE_ICD',
    CONCAT(
      'Procedure [ICD', CAST(pr.icd_version AS STRING),
      ' #', CAST(pr.seq_num AS STRING), ']: ',
      pr.icd_code, ' – ', dipr.long_title
    ),
    NULL, NULL, NULL, 'date_only'
  FROM `{HOSP_DS}.procedures_icd` pr
  JOIN `{HOSP_DS}.d_icd_procedures` dipr
    ON pr.icd_code = dipr.icd_code AND pr.icd_version = dipr.icd_version
  WHERE pr.hadm_id = @hadm_id

  UNION ALL

  -- HOSP: Hospital discharge
  SELECT
    adm.hadm_id,
    adm.dischtime, 'HOSP', 'DISCHARGE',
    CONCAT('Discharged | Destination: ',
           COALESCE(adm.discharge_location, 'unknown')),
    NULL, NULL, NULL, 'exact'
  FROM adm
  WHERE adm.dischtime IS NOT NULL

  -- ── ICU MODULE ────────────────────────────────────────────────────────────

  UNION ALL

  -- ICU: Vital signs – filtered to key items to avoid full table scan
  SELECT
    c.hadm_id,
    c.charttime, 'ICU', 'ICU_VITAL',
    CONCAT(
      di.label, ': ',
      COALESCE(c.value, CAST(c.valuenum AS STRING), 'no value'),
      COALESCE(CONCAT(' ', c.valueuom), '')
    ),
    CAST(c.valuenum AS STRING),
    c.valueuom,
    CAST(c.warning AS STRING),
    'exact'
  FROM `{ICU_DS}.chartevents` c
  JOIN `{ICU_DS}.d_items` di ON c.itemid = di.itemid
  WHERE c.hadm_id = @hadm_id
    AND c.itemid IN ({VITAL_IDS_STR})

  UNION ALL

  -- ICU: IV inputs – fluids and medications administered in ICU
  SELECT
    ie.hadm_id,
    ie.starttime, 'ICU', 'ICU_INPUT',
    CONCAT(
      di.label, ': ',
      CAST(ROUND(ie.amount, 2) AS STRING), ' ', ie.amountuom,
      COALESCE(CONCAT(' @ ', CAST(ROUND(ie.rate, 2) AS STRING),
                      ' ', ie.rateuom), '')
    ),
    CAST(ie.amount AS STRING),
    ie.amountuom,
    NULL,
    'exact'
  FROM `{ICU_DS}.inputevents` ie
  JOIN `{ICU_DS}.d_items` di ON ie.itemid = di.itemid
  WHERE ie.hadm_id = @hadm_id

  UNION ALL

  -- ICU: Outputs – urine, drains, etc.
  SELECT
    oe.hadm_id,
    oe.charttime, 'ICU', 'ICU_OUTPUT',
    CONCAT(di.label, ': ', CAST(oe.value AS STRING), ' ', oe.valueuom),
    CAST(oe.value AS STRING),
    oe.valueuom,
    NULL,
    'exact'
  FROM `{ICU_DS}.outputevents` oe
  JOIN `{ICU_DS}.d_items` di ON oe.itemid = di.itemid
  WHERE oe.hadm_id = @hadm_id

  UNION ALL

  -- ICU: Procedures – IV lines, arterial lines, OR transport, etc.
  SELECT
    pe.hadm_id,
    pe.starttime, 'ICU', 'ICU_PROCEDURE',
    CONCAT(
      'ICU procedure: ', di.label,
      COALESCE(CONCAT(' (', CAST(pe.value AS STRING),
                      ' ', COALESCE(pe.valueuom, ''), ')'), '')
    ),
    CAST(pe.value AS STRING),
    pe.valueuom,
    NULL,
    'exact'
  FROM `{ICU_DS}.procedureevents` pe
  JOIN `{ICU_DS}.d_items` di ON pe.itemid = di.itemid
  WHERE pe.hadm_id = @hadm_id

  {notes_sql}

)

-- ── Final output ──────────────────────────────────────────────────────────────
-- elapsed_hours is NULL for date_only records (only a calendar date is known).
-- anchored records get elapsed_hours computed from their anchor time, but
-- time_precision='anchored' signals the timestamp is not independently recorded.
SELECT
  adm.subject_id,
  adm.hadm_id,
  ae.event_time,
  CASE WHEN ae.time_precision = 'date_only' THEN NULL
       ELSE ROUND(DATETIME_DIFF(ae.event_time, t0_cte.t0, SECOND) / 3600.0, 3)
  END                AS elapsed_hours,
  ae.time_precision,
  ae.source,
  ae.event_type,
  ae.description,
  ae.value,
  ae.unit,
  ae.flag

FROM all_events ae
JOIN adm ON ae.hadm_id = adm.hadm_id
JOIN t0_cte ON ae.hadm_id = t0_cte.hadm_id
WHERE ae.event_time IS NOT NULL
ORDER BY ae.event_time, ae.source, ae.event_type
"""


def _parse_history(text: str) -> str:
    """Extract History of Present Illness section."""
    text = text.replace("\n", " ")
    success = False
    i = 0
    pe_strings = [
        "physical exam:",
        "physical examination:",
        "physical ___:",
        "pe:",
        "pe ___:",
        "(?:pertinent|___) results:",
        "hospital course:",
    ]
    while not success and i < len(pe_strings):
        regex = re.compile(
            rf"(?:history|___) of present(?:ing)? illness:.*?{pe_strings[i]}",
            re.IGNORECASE | re.DOTALL,
        )
        m = regex.search(text)
        if m:
            text = m.group(0)
            success = True
        i += 1

    if not success:
        return ""

    text = re.sub(re.compile("history of present(?:ing)? illness:", re.IGNORECASE), "", text)

    for pe_str in pe_strings:
        text = re.sub(re.compile(pe_str, re.IGNORECASE), "", text)

    return text.strip()


def _parse_physical_exam(text: str) -> str:
    """Extract Physical Examination section."""
    text = text.replace("\n", " ")
    success = False
    i = 0
    pe_strings = [
        "physical exam:",
        "physical examination:",
        "physical ___:",
        "pe:",
        "pe ___:",
        "pertinent results:",
    ]
    while not success and i < len(pe_strings):
        terminal_str = "pertinent results:"
        if terminal_str not in text.lower():
            terminal_str = "brief hospital course:"
        regex = re.compile(
            rf"{pe_strings[i]}.*?{terminal_str}", re.IGNORECASE | re.DOTALL
        )
        m = regex.search(text)
        if m:
            text = m.group(0)
            success = True
        i += 1

    if not success:
        return ""

    for pe_str in pe_strings:
        text = re.sub(re.compile(pe_str, re.IGNORECASE), "", text)

    text = re.sub(re.compile("pertinent results:", re.IGNORECASE), "", text)
    text = re.sub(re.compile("brief hospital course:", re.IGNORECASE), "", text)

    text = re.sub(re.compile("at discharge.*", re.IGNORECASE), "", text)
    text = re.sub(re.compile("upon discharge.*", re.IGNORECASE), "", text)
    text = re.sub(re.compile("on discharge.*", re.IGNORECASE), "", text)
    text = re.sub(re.compile("discharge.*", re.IGNORECASE), "", text)

    return text.strip()


def _parse_discharge_diagnosis(text: str) -> str:
    """Extract free-text Discharge Diagnosis section."""
    tl = text.lower()
    start = 0
    for hdr in ["discharge diagnosis:", "___ diagnosis:"]:
        pos = tl.rfind(hdr)
        if pos != -1:
            start = max(start, pos + len(hdr))
    if not start:
        pos = tl.rfind("\n___:")
        if pos != -1:
            start = pos
        else:
            return ""
    end = 0
    for hdr in ["discharge condition:", "___ condition:", "condition:",
                "procedure:", "procedures:", "invasive procedure on this admission:"]:
        pos = tl.rfind(hdr)
        if pos != -1 and pos > start:
            end = max(end, pos)
            break
    if not end:
        return ""
    return text[start:end].strip()


def parse_discharge_note_sections(
    hadm_id: int,
    text: str,
    note_time: "pd.Timestamp",
    t0: "pd.Timestamp",
    subject_id=None,
) -> list[dict]:
    """
    Parse HPI / Physical Exam / free-text diagnosis from a discharge note and
    return them as timeline row dicts.
    """
    sections = {
        "DISCHARGE_HPI": _parse_history(text),
        "DISCHARGE_PE": _parse_physical_exam(text),
        "DISCHARGE_FREETEXTDX": _parse_discharge_diagnosis(text),
    }

    rows = []
    for event_type, content in sections.items():
        if not content:
            continue
        event_time_anchor = t0 if event_type in ["DISCHARGE_HPI", "DISCHARGE_PE"] else note_time

        elapsed = (
            round((event_time_anchor - t0).total_seconds() / 3600, 3)
            if pd.notna(event_time_anchor) and pd.notna(t0)
            else None
        )
        rows.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "event_time": event_time_anchor,
            "elapsed_hours": elapsed,
            "time_precision": "anchored",
            "source": "NOTE",
            "event_type": event_type,
            "description": content,
            "value": None,
            "unit": None,
            "flag": None,
        })

    return rows


def get_timeline(
    hadm_id: int,
    project: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch the full timeline for a single hadm_id from BigQuery.

    Parameters
    ----------
    hadm_id        : Hospital admission ID to reconstruct.
    project        : GCP billing project. If None, uses the default configured

    Returns
    -------
    pd.DataFrame with columns:
      subject_id, hadm_id, event_time, elapsed_hours, time_precision,
      source, event_type, description, value, unit, flag
    """
    client = bigquery.Client(project=project)

    sql = build_query()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("hadm_id", "INT64", hadm_id),
        ]
    )

    print(f"Querying BigQuery for hadm_id={hadm_id} ...")
    job = client.query(sql, job_config=job_config)
    df = job.to_dataframe()

    if df.empty:
        print(f"WARNING: No rows returned. Check that hadm_id={hadm_id} exists.")
        return df

    bytes_billed = job.total_bytes_billed or 0
    print(f"  Rows returned : {len(df):,}")
    print(f"  Bytes billed  : {bytes_billed / 1e6:.1f} MB")

    # t0 = earliest non-date_only event time (mirrors SQL t0_cte logic)
    t0 = df.loc[df["time_precision"] != "date_only", "event_time"].min()
    subject_id = df["subject_id"].iloc[0]

    print("  Parsing discharge note sections (HPI / PE / diagnosis) ...")
    note_query = f"""
    SELECT charttime, text
    FROM `{NOTE_DS}.discharge`
    WHERE hadm_id = @hadm_id
    LIMIT 1
    """
    note_job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("hadm_id", "INT64", hadm_id)]
    )
    note_df = client.query(note_query, job_config=note_job_config).to_dataframe()
    if not note_df.empty and note_df["text"].iloc[0]:
        note_rows = parse_discharge_note_sections(
            hadm_id=hadm_id,
            text=note_df["text"].iloc[0],
            note_time=note_df["charttime"].iloc[0],
            t0=t0,
            subject_id=subject_id,
        )
        if note_rows:
            df = pd.concat([df, pd.DataFrame(note_rows)], ignore_index=True)
            print(f"  Added {len(note_rows)} discharge note section rows.")

    df = sort_timeline(df)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the timeline."""
    if df.empty:
        return

    sid  = df["subject_id"].iloc[0]
    hid  = df["hadm_id"].iloc[0]
    t0   = df["event_time"].min()
    tend = df["event_time"].max()
    span = df["elapsed_hours"].max()

    print(f"\n{'='*70}")
    print(f"Timeline: subject_id={sid}  hadm_id={hid}")
    print(f"  From : {t0}  →  {tend}")
    print(f"  Span : {span:.1f} hours ({span/24:.1f} days)")
    print(f"  Total events: {len(df):,}")
    print(f"\nEvent counts by source:")
    counts = df.groupby(["source", "event_type"]).size().reset_index(name="n")
    for _, row in counts.sort_values("n", ascending=False).iterrows():
        print(f"  {row['source']:6s} / {row['event_type']:20s} : {row['n']:>5,}")
    print("="*70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a MIMIC-IV patient HADM timeline from BigQuery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hadm_id", type=int, required=True,
        help="Hospital admission ID to reconstruct",
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="GCP billing project ID (uses default if not set)",
    )
    parser.add_argument(
        "--print-sql", action="store_true",
        help="Print the generated SQL and exit",
    )
    args = parser.parse_args()

    if args.print_sql:
        sql = build_query()
        print(sql.replace("@hadm_id", str(args.hadm_id)))
        return

    df = get_timeline(
        hadm_id=args.hadm_id,
        project=args.project,
    )

    print_summary(df)

    out_file = f"timeline_{args.hadm_id}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
