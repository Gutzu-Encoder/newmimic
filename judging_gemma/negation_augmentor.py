"""Negation Augmentor — adds 3 new negation types to neweraugmented_mimic.csv (C3).

Types:
  negated_hx       — negate key findings in Past Medical History section
  negated_ruled_out — convert active dx mentions to "ruled out" phrasing in hospital course
  negated_hedge    — convert definitive imaging findings to hedged language

Appends directly to neweraugmented_mimic.csv in-place.
"""

import re, pandas as pd
from pathlib import Path

MIMIC_PATH = Path(r"C:\Users\USER\Downloads\neweraugmented_mimic_admission_only.csv")

# ── PMH section pattern ──────────────────────────────────────────────────────
_PMH_START = re.compile(r'Past Medical History\s*:', re.IGNORECASE)
_PMH_END   = re.compile(r'(?:Social History|Family History|Physical Exam|Pertinent Results|Imaging|Brief Hospital Course)', re.IGNORECASE)

# ── Hospital Course section pattern ──────────────────────────────────────────
_COURSE_START = re.compile(r'Brief Hospital Course', re.IGNORECASE)
_COURSE_END   = re.compile(r'(?:Medications on Admission|Discharge Medications|Discharge Disposition|Followup Instructions|Please return)', re.IGNORECASE)

# ── Imaging section pattern ──────────────────────────────────────────────────
_IMAGING_START = re.compile(r'(?:Imaging|Pertinent Results)\s*:', re.IGNORECASE)
_IMAGING_END   = re.compile(r'(?:Brief Hospital Course|Medications on Admission)', re.IGNORECASE)


def _extract_section(text: str, start_re, end_re) -> tuple[str, int, int]:
    """Return (section_text, start_idx, end_idx) or ('', -1, -1) if not found."""
    m_start = start_re.search(text)
    if not m_start:
        return '', -1, -1
    s = m_start.start()
    m_end = end_re.search(text, s + 1)
    e = m_end.start() if m_end else len(text)
    return text[s:e], s, e


def _negated_hx(text: str) -> str:
    """Negate key diagnoses in Past Medical History section.

    Rules:
    - Prefix each major diagnosis with negation language
    - Keep comorbidities but cast doubt on them
    """
    section, s, e = _extract_section(text, _PMH_START, _PMH_END)
    if not section:
        return text

    def _negate_item(m: re.Match) -> str:
        num = m.group(1)
        content = m.group(2).strip()
        major_keywords = ['metastatic', 'cancer', 'malignant', 'effusion', 'septic',
                          'abscess', 'infection', 'obstruction', 'hemorrhage', 'bleed',
                          'failure', 'syndrome']
        is_major = any(k in content.lower() for k in major_keywords)
        if is_major:
            return f"{num}) No known history of {content} — patient denies prior diagnosis"
        else:
            return f"{num}) History of {content} (patient reports uncertain)"

    negated_section = re.sub(
        r'(\d+)\)\s+([^\n]+)',
        _negate_item,
        section
    )
    return text[:s] + negated_section + text[e:]


def _negated_ruled_out(text: str) -> str:
    """Convert active diagnosis phrasing in hospital course to "ruled out" language."""
    section, s, e = _extract_section(text, _COURSE_START, _COURSE_END)
    if not section:
        return text

    replacements = [
        (r'This (?:likely )?represented (?:an element of )?([^\n\.]+)',
         r'\1 was considered but ultimately ruled out'),
        (r'This was (?:thought to be|believed to be) due to ([^\n\.]+)',
         r'\1 was initially suspected but later ruled out'),
        (r'Likely etiology of ([^\n\.]+) is ([^\n\.]+)',
         r'\1 — \2 was considered but ruled out by further workup'),
        (r'([A-Z][a-zA-Z\s]+) likely represents (?:a )?([^\n\.]+)',
         r'\1 — \2 was considered in differential but ruled out'),
    ]

    negated_section = section
    for pattern, repl in replacements:
        negated_section = re.sub(pattern, repl, negated_section, flags=re.IGNORECASE)

    return text[:s] + negated_section + text[e:]


def _negated_hedge(text: str) -> str:
    """Convert definitive imaging findings to hedged language."""
    section, s, e = _extract_section(text, _IMAGING_START, _IMAGING_END)
    if not section:
        section = text
        s, e = 0, len(text)

    hedges = [
        (r'\bconsistent with\b', 'possibly consistent with'),
        (r'\bmost likely represents\b', 'may represent'),
        (r'\blikely represents\b', 'could represent'),
        (r'\brepresents\b', 'may represent'),
        (r'\bindicates\b', 'could indicate'),
        (r'\bsuggestive of\b', 'possibly suggestive of'),
        (r'\bdiagnostic of\b', 'possibly consistent with'),
        (r'\bNo significant change\b', 'No definite significant change'),
    ]

    hedged_section = section
    for pattern, repl in hedges:
        hedged_section = re.sub(pattern, repl, hedged_section, flags=re.IGNORECASE)

    hedged_section = re.sub(
        r'\bhighly suggestive of\b', 'vaguely suggestive of',
        hedged_section, flags=re.IGNORECASE
    )

    return text[:s] + hedged_section + text[e:]


def _create_augmentation_rows(df_base: pd.DataFrame) -> pd.DataFrame:
    """Generate 3 new rows per HADM_ID from existing rows (prefer negated as source)."""
    rows = []
    for hadm_id, grp in df_base.groupby('HADM_ID'):
        src = grp[grp['augmentation_type'] == 'negated']
        if src.empty:
            src = grp.iloc[[0]]
        src_row = src.iloc[0].to_dict()
        base_text = src_row['cleaned_text']

        for aug_type, transform in [
            ('negated_hx', _negated_hx),
            ('negated_ruled_out', _negated_ruled_out),
            ('negated_hedge', _negated_hedge),
        ]:
            new_row = dict(src_row)
            new_row['augmentation_type'] = aug_type
            new_row['cleaned_text'] = transform(base_text)
            new_row['is_augmented'] = True
            rows.append(new_row)

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(MIMIC_PATH)
    print(f"Loaded {len(df)} rows from {MIMIC_PATH}")

    new_rows = _create_augmentation_rows(df)
    print(f"Generated {len(new_rows)} new augmentation rows")

    df_out = pd.concat([df, new_rows], ignore_index=True)
    df_out.to_csv(MIMIC_PATH, index=False)
    print(f"Saved {len(df_out)} rows back to {MIMIC_PATH}")

    print("\naugmentation_type distribution:")
    print(df_out['augmentation_type'].value_counts())


if __name__ == '__main__':
    main()
