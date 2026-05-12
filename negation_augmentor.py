"""
negation_augmentor.py

Adds 3 new negation augmentation types to the MIMIC discharge note dataset,
targeting different clinical sections than the existing "negated" type (which
only flips HPI denial statements).

Each type targets a different section because each section encodes different
clinical information that models rely on for diagnosis:

  negated_hx        → Past Medical History
                       Tests: does the model use PMH to anchor its diagnosis?
                       Real-world analogue: EMR carry-forward errors, wrong PMH copied
                       from prior admission.

  negated_ruled_out → Brief Hospital Course
                       Tests: does the model read the hospital course to shortcut the
                       answer rather than reasoning from clinical evidence?
                       Real-world analogue: model reads discharge plan instead of
                       doing true clinical reasoning from note content.

  negated_hedge     → Imaging / Pertinent Results
                       Tests: does the model correctly interpret imaging certainty
                       modifiers? Flips definitive findings to absent/uncertain and
                       negated findings to present.
                       Real-world analogue: radiology report certainty modifiers
                       ("consistent with", "cannot exclude") carry diagnostic weight
                       that models frequently ignore.

Usage:
    python negation_augmentor.py

Output:
    C:/Users/USER/Downloads/neweraugmented_mimic_extended.csv
    — same schema as neweraugmented_mimic.csv but with 3 new augmentation_type values
    — 10 HADMs × 3 new types = 30 new rows appended to the original
"""

import re
import pandas as pd
from pathlib import Path

INPUT_CSV  = Path(r"C:\Users\USER\Downloads\neweraugmented_mimic.csv")
OUTPUT_CSV = Path(r"C:\Users\USER\Downloads\neweraugmented_mimic_extended.csv")


# ── Section boundary definitions ──────────────────────────────────────────────
# Each tuple: (header_regex, [end_regex, ...])
# We slice text from the first char of the header match to the first end match.

SECTION_BOUNDS = {
    "pmh": (
        r'(?:Past Medical History|PAST MEDICAL HISTORY|PMH)\s*:',
        [
            r'\n(?:Social History|SOCIAL HISTORY)',
            r'\n(?:Family History|FAMILY HISTORY)',
            r'\n(?:Allergies|ALLERGIES)',
            r'\n(?:Physical Exam|PHYSICAL EXAM)',
            r'\n(?:Medications|MEDICATIONS)\s+(?:on|ON)',
        ]
    ),
    "course": (
        r'(?:Brief Hospital Course|BRIEF HOSPITAL COURSE|HOSPITAL COURSE)\s*:',
        [
            r'\nMedications\s+on\s+(?:Admission|Discharge)',
            r'\nMEDICATIONS\s+ON',
            r'\nDischarge\s+(?:Medications|Disposition|Diagnoses|Condition)',
            r'\nDISCHARGE\s+(?:MEDICATIONS|DIAGNOSES|CONDITION)',
        ]
    ),
    "imaging": (
        r'(?:Pertinent Results|PERTINENT RESULTS|Imaging)\s*:',
        [
            r'\n(?:Brief Hospital Course|BRIEF HOSPITAL COURSE|HOSPITAL COURSE)',
            r'\n(?:Discharge|DISCHARGE)',
        ]
    ),
}


def find_section(text: str, section_key: str) -> tuple[int, int]:
    """Return (start, end) character indices for the named section. (-1,-1) if not found."""
    header_pat, end_pats = SECTION_BOUNDS[section_key]
    hm = re.search(header_pat, text, re.IGNORECASE)
    if not hm:
        return -1, -1
    start = hm.start()
    end = len(text)
    for ep in end_pats:
        em = re.search(ep, text[hm.end():], re.IGNORECASE)
        if em:
            end = min(end, hm.end() + em.start())
    return start, end


def replace_section(text: str, start: int, end: int, new_section: str) -> str:
    return text[:start] + new_section + text[end:]


# ══════════════════════════════════════════════════════════════════════════════
# TYPE 1 — negated_hx
# Target: Past Medical History section
# Strategy: Prepend "No prior history of" to each numbered/bulleted condition.
#           Flip "status post X" → "no prior X".
#           Flip "history of X" → "no history of X".
#
# Before:
#   1) Metastatic breast cancer: T2-NO, ER/PR+...
#   2) Malignant pleural effusion s/p pleurex catheter placed removed
#   3) Asthma
#
# After:
#   1) No prior history of metastatic breast cancer: T2-NO, ER/PR+...
#   2) No prior history of malignant pleural effusion s/p pleurex catheter placed removed
#   3) No prior history of asthma
# ══════════════════════════════════════════════════════════════════════════════

PMH_RULES = [
    # numbered list "1) Condition" or "1. Condition"
    (
        r'(\b\d+[).]\s+)([A-Z][a-z])',
        lambda m: m.group(1) + 'No prior history of ' + m.group(2)
    ),
    # "Status post X" / "s/p X"
    (
        r'\b[Ss]tatus\s+post\b',
        lambda m: 'No prior'
    ),
    (
        r'\bs/p\b',
        lambda m: 'no prior'
    ),
    # "history of X" → "no history of X"
    (
        r'\b([Hh])istory of\b',
        lambda m: m.group(1) + 'o history of'   # H→ 'Ho history' is wrong; do this:
    ),
]

# Cleaner approach: apply each as a re.sub with a string replacement
PMH_SUB_RULES = [
    # numbered list  "1) X" or "1. X"
    (r'(\b\d+[).]\s+)([A-Z])', r'\1No prior history of \2'),
    # status post
    (r'\bStatus post\b', 'No prior'),
    (r'\bstatus post\b', 'no prior'),
    (r'\bS/[Pp]\b',      'no prior'),
    (r'\bs/[Pp]\b',      'no prior'),
    # "History of" → "No history of"
    (r'\bHistory of\b',  'No history of'),
    (r'\bhistory of\b',  'no history of'),
    # "Diagnosed with X" → "No diagnosis of X"
    (r'\b[Dd]iagnosed with\b', 'No diagnosis of'),
    # "Known X" → "No known X" (if not already negated)
    (r'\b([Kk])nown (?!drug|allerg)', r'\1No known '),
]


def negate_pmh(text: str) -> str:
    start, end = find_section(text, "pmh")
    if start == -1:
        return text   # section not found, return unchanged
    section = text[start:end]
    for pattern, replacement in PMH_SUB_RULES:
        section = re.sub(pattern, replacement, section)
    return replace_section(text, start, end, section)


# ══════════════════════════════════════════════════════════════════════════════
# TYPE 2 — negated_ruled_out
# Target: Brief Hospital Course section
# Strategy:
#   (a) Problem-based notes (# headers): flip each problem conclusion to ruled-out.
#       "#HYPOXIA The patient had a known pleural effusion..." → "#HYPOXIA was ruled out."
#   (b) Systems-based notes (A. Respiratory, B. Cardiovascular, etc.):
#       flip system conclusions to normal/absent.
#   (c) Final diagnosis statements: negate.
#
# Before:
#   #HYPOXIA The patient had a known chronic R pleural effusion...
#   #Hypercalcemia This was thought to be due to malignancy.
#   She was given pamidronate 90 mg IV, with some response in her calcium levels.
#
# After:
#   #HYPOXIA No hypoxia was identified during this admission.
#   #Hypercalcemia was not identified. Calcium levels remained within normal limits.
#   She did not require pamidronate.
# ══════════════════════════════════════════════════════════════════════════════

COURSE_CONFIRMED_PHRASES = [
    # "The patient had X" → "The patient did not have X"
    (r'\bThe patient had\b',         'The patient did not have'),
    (r'\bThe patient was found to have\b', 'The patient was found not to have'),
    # "She/He had X" → "She/He did not have X"
    (r'\b([Ss]he|[Hh]e) had\b',     r'\1 did not have'),
    # "X was identified/confirmed/demonstrated"
    (r'\bwas identified\b',          'was not identified'),
    (r'\bwas confirmed\b',           'was not confirmed'),
    (r'\bwas demonstrated\b',        'was not demonstrated'),
    (r'\bwas noted\b',               'was not noted'),
    (r'\bwas found\b',               'was not found'),
    # "likely X" → "unlikely X"
    (r'\b([Ll])ikely\b(?! not)',     r'\1ikely not'),
    # "represents X" → "does not represent X"
    (r'\b([Rr])epresents?\b',        r'does not represent'),
    # "was treated for X" → "did not require treatment for X"
    (r'\bwas treated for\b',         'did not require treatment for'),
    # "given X" (medication administered) → "did not receive X"
    (r'\bwas given\b',               'was not given'),
    (r'\breceived\b',                'did not receive'),
    # "continued on X" → "X was not continued"
    (r'\bwas continued on\b',        'was not continued on'),
    # "due to malignancy/infection/etc." → "not attributed to"
    (r'\bdue to\b',                  'not attributed to'),
    (r'\bsecondary to\b',            'not secondary to'),
]


def negate_hospital_course(text: str) -> str:
    start, end = find_section(text, "course")
    if start == -1:
        return text
    section = text[start:end]

    # (a) Collapse problem-based # headers to single-line ruled-out statements.
    # Pattern: ". #PROBLEM_NAME rest-of-paragraph" (problem paragraphs end at next ". #")
    def collapse_problem(m):
        header = m.group(1).strip()   # e.g. "HYPOXIA" or "Breast CA"
        return f'\n. #{header} was ruled out and not identified during this admission.\n'

    section = re.sub(
        r'\.\s+#([A-Z][^\n.]{1,60})\n(.+?)(?=\n\.\s+#|\Z)',
        collapse_problem,
        section,
        flags=re.DOTALL
    )

    # (b) Apply phrase-level negation rules to remaining free text
    for pattern, replacement in COURSE_CONFIRMED_PHRASES:
        section = re.sub(pattern, replacement, section)

    return replace_section(text, start, end, section)


# ══════════════════════════════════════════════════════════════════════════════
# TYPE 3 — negated_hedge
# Target: Imaging sub-section within Pertinent Results
# Strategy:
#   (a) Definitive positive findings → absent/not demonstrated.
#       "1. Chronic pulmonary embolism of the left main PA" → "1. No pulmonary embolism identified"
#       "consistent with omental carcinomatosis" → "inconsistent with omental carcinomatosis"
#   (b) Established negations → positive.
#       "No acute PE identified" → "Acute PE identified"
#       "No significant change in hepatic metastases" → "Interval progression of hepatic metastases"
#   (c) Hedged language → definitive absence.
#       "probably atelectasis" → "consolidation is not atelectasis"
#       "possibly indicating lymphangitic spread" → "no evidence of lymphangitic spread"
#
# Before (HADM 197345 CT Torso):
#   1. Chronic pulmonary embolism of the left main pulmonary artery...
#   2. New moderate left pleural effusion...
#   5. Interval in now moderate-to-severe ascites and increased omental soft tissue
#      consistent with omental carcinomatosis.
#   6. No significant change in innumerable hepatic metastases.
#
# After:
#   1. No pulmonary embolism identified in the left main pulmonary artery...
#   2. No left pleural effusion...
#   5. No ascites. Omental soft tissue inconsistent with omental carcinomatosis.
#   6. Interval progression of innumerable hepatic metastases.
# ══════════════════════════════════════════════════════════════════════════════

IMAGING_RULES = [
    # ── Flip explicit negations to positive ───────────────────────────────────
    # "No acute X identified" → "Acute X identified"
    (r'\bNo acute ([a-z])',           r'Acute \1'),
    # "No X identified/seen/noted/present" → "X identified"
    (r'\bNo ([a-z][a-z ]{2,40}) (?:identified|seen|noted|visualized|present|demonstrated)\b',
     r'\1 identified'),
    # "No significant change in X" → "Interval progression of X"
    (r'\bNo significant change in ([a-z][^\.\n]{3,60})',
     r'Interval progression of \1'),
    # "No evidence of X" → "Evidence of X"
    (r'\bNo evidence of ([a-z][^\.\n]{3,60})',
     r'Evidence of \1'),
    # "cannot be excluded" → "confirmed"
    (r'cannot be excluded',          'confirmed'),
    (r'can not be excluded',         'confirmed'),

    # ── Flip definitive positives to absent ───────────────────────────────────
    # "New X" → "No new X"  (new finding present → new finding absent)
    (r'\bNew ([a-z])',               r'No new \1'),
    # "Chronic X of the..." → "No chronic X of the..."
    (r'\bChronic ([a-z][a-z ]{2,40}) of\b',
     r'No chronic \1 of'),
    # "consistent with X" → "inconsistent with X"
    (r'\bconsistent with\b',         'inconsistent with'),
    # "X is present" → "X is absent"
    (r'\bis present\b',              'is absent'),
    # "demonstrates X" / "shows X" → "does not demonstrate X"
    (r'\b([Dd])emonstrates? ([a-z])',r'\1oes not demonstrate \2'),
    (r'\b([Ss])hows? ([a-z])',       r'\1hows no \2'),
    # "moderate/severe/mild X" → "no X"
    (r'\b(?:moderate|mild|severe|large|small)-?to-?(?:moderate|severe)?\s+([a-z][a-z ]{2,40})',
     r'no \1'),

    # ── Flip hedged language to definitive absence ────────────────────────────
    # "probably X" → "not X"
    (r'\bprobably ([a-z][^\.,\n]{2,60})',   r'not \1'),
    # "possibly X" / "possibly indicating X" → "no evidence of X"
    (r'\bpossibly (?:indicating )?([a-z][^\.,\n]{2,60})',
     r'no evidence of \1'),
    # "likely X" → "unlikely to represent X"
    (r'\blikely ([a-z][^\.,\n]{2,50})',     r'unlikely to represent \1'),
    # "questionable X" → "no X"
    (r'\bquestionable ([a-z][^\.,\n]{2,50})',
     r'no \1'),
]


def negate_imaging(text: str) -> str:
    start, end = find_section(text, "imaging")
    if start == -1:
        return text
    section = text[start:end]

    # Only apply within the imaging sub-section (after "Imaging:" or "CT" or "Echo" lines)
    # Find first imaging keyword and apply rules from there
    img_start = re.search(
        r'\b(?:Imaging|CT\b|Echo|CXR|Chest\s*X.?[Rr]ay|MRI|Ultrasound|KUB)\b',
        section, re.IGNORECASE
    )
    if img_start:
        prefix = section[:img_start.start()]
        img_text = section[img_start.start():]
        for pattern, replacement in IMAGING_RULES:
            img_text = re.sub(pattern, replacement, img_text)
        section = prefix + img_text
    else:
        # No imaging keyword found — apply rules to full Pertinent Results section
        for pattern, replacement in IMAGING_RULES:
            section = re.sub(pattern, replacement, section)

    return replace_section(text, start, end, section)


# ══════════════════════════════════════════════════════════════════════════════
# Main: generate 3 new augmentation rows per HADM and write output
# ══════════════════════════════════════════════════════════════════════════════

AUGMENTATION_FNS = {
    'negated_hx':         negate_pmh,
    'negated_ruled_out':  negate_hospital_course,
    'negated_hedge':      negate_imaging,
}


def self_test():
    """Spot-check each function on HADM 197345 text before running the full CSV."""
    sample = (
        "Past Medical History: 1) Metastatic breast cancer: T2-NO, diagnosed 2010. "
        "Status post CMF. 2) Malignant pleural effusion s/p pleurex catheter. 3) Asthma.\n"
        "Social History: lives with husband.\n"
        "Brief Hospital Course: . #HYPOXIA The patient had a known chronic R pleural effusion "
        "with SOB. She was treated for hypoxia.\n"
        ". #Hypercalcemia This was thought to be due to malignancy. She was given pamidronate.\n"
        "Medications on Admission: Lovenox.\n"
        "Pertinent Results: Imaging: CT Torso: 1. Chronic pulmonary embolism of the left main PA. "
        "2. New moderate left pleural effusion. "
        "3. No acute PE identified. "
        "5. Ascites consistent with omental carcinomatosis. "
        "6. No significant change in hepatic metastases. "
        "7. Probably atelectasis.\n"
        "Brief Hospital Course: placeholder"
    )

    print("=" * 60)
    print("SELF-TEST")
    print("=" * 60)

    for aug_type, fn in AUGMENTATION_FNS.items():
        result = fn(sample)
        print(f"\n── {aug_type} ──")
        # Show only the modified section for brevity
        if aug_type == 'negated_hx':
            m = re.search(r'Past Medical History.+?Social History', result, re.DOTALL)
        elif aug_type == 'negated_ruled_out':
            m = re.search(r'Brief Hospital Course.+?Medications on Admission', result, re.DOTALL)
        else:
            m = re.search(r'Pertinent Results.+', result, re.DOTALL)
        print(m.group(0)[:600] if m else result[:600])


def main():
    self_test()
    print("\n" + "=" * 60)
    print("PROCESSING CSV")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)

    # Take one representative row per HADM (base/unaugmented text from any augmentation type)
    base_rows = df.drop_duplicates(subset='HADM_ID', keep='first').copy()
    print(f"Loaded {len(df)} rows. Using 1 base row per HADM ({len(base_rows)} HADMs).")

    new_rows = []
    for _, row in base_rows.iterrows():
        original_text = row['cleaned_text']
        for aug_type, fn in AUGMENTATION_FNS.items():
            new_text = fn(original_text)
            changed = new_text != original_text
            new_row = row.copy()
            new_row['cleaned_text'] = new_text
            new_row['augmentation_type'] = aug_type
            new_row['is_augmented'] = changed
            new_rows.append(new_row)
            status = "CHANGED" if changed else "UNCHANGED (section not found)"
            print(f"  HADM {row['HADM_ID']} | {aug_type}: {status}")

    df_new = pd.DataFrame(new_rows)
    df_extended = pd.concat([df, df_new], ignore_index=True)
    df_extended.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. {len(df_new)} new rows added.")
    print(f"Output: {OUTPUT_CSV}")
    print(f"\nAugmentation type counts:")
    print(df_extended['augmentation_type'].value_counts().to_string())

    # Spot check: show one changed section per type
    for aug_type in AUGMENTATION_FNS:
        changed = df_new[(df_new['augmentation_type'] == aug_type) & df_new['is_augmented']]
        if changed.empty:
            print(f"\n  WARNING: {aug_type} — no rows were modified. Check section headers.")
            continue
        sample_hadm = changed.iloc[0]
        base_text   = base_rows[base_rows['HADM_ID'] == sample_hadm['HADM_ID']].iloc[0]['cleaned_text']
        new_text    = sample_hadm['cleaned_text']
        # Show first diff
        for i, (a, b) in enumerate(zip(base_text.split('.'), new_text.split('.'))):
            if a != b:
                print(f"\n  {aug_type} (HADM {sample_hadm['HADM_ID']}) first change at sentence {i}:")
                print(f"    BEFORE: {a.strip()[:120]}")
                print(f"    AFTER:  {b.strip()[:120]}")
                break


if __name__ == '__main__':
    main()