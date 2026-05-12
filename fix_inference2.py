"""Fix MedGemma and Qwen inference notebooks via regex on joined source."""
import json, re
from pathlib import Path

DL = Path(r"C:\Users\USER\Downloads\judging_gemma")

# Regex patterns that work regardless of blank lines between source array elements
PATTERNS = [
    # 1. DATA_PATH
    (
        r'DATA_PATH\s*=\s*"/content/drive/MyDrive/neweraugmented_mimic_noleak\.csv"',
        'DATA_PATH = "/content/drive/MyDrive/neweraugmented_mimic_admission_only.csv"'
    ),
    (
        r'DATA_PATH\s*=\s*"/content/drive/MyDrive/neweraugmented_mimic\.csv"',
        'DATA_PATH = "/content/drive/MyDrive/neweraugmented_mimic_admission_only.csv"'
    ),
    # 2. Negated lookup → 3 lookups
    (
        r"negated_text\s*=\s*df\[\s*\(df\['HADM_ID'\]\s*==\s*hadm_id\)\s*&\s*\(df\['augmentation_type'\]\s*==\s*'negated'\)\s*\]\['cleaned_text'\]\.values\[0\]",
        """negated_hx_text = df[
        (df['HADM_ID'] == hadm_id) &
        (df['augmentation_type'] == 'negated_hx')
    ]['cleaned_text'].values[0]

    negated_ruled_text = df[
        (df['HADM_ID'] == hadm_id) &
        (df['augmentation_type'] == 'negated_ruled_out')
    ]['cleaned_text'].values[0]

    negated_hedge_text = df[
        (df['HADM_ID'] == hadm_id) &
        (df['augmentation_type'] == 'negated_hedge')
    ]['cleaned_text'].values[0]"""
    ),
    # 3. conditions list A (None tuples) - with flexible whitespace
    (
        r'conditions\s*=\s*\[\s*\("0-shot",\s*test_row\[\'cleaned_text\'\],\s*None\),\s*\("1-shot",\s*test_row\[\'cleaned_text\'\],\s*EXPERT_EXAMPLE\),\s*\("counterfactual",\s*test_row\[\'cleaned_text\'\],\s*COUNTERFACTUAL_EXAMPLE\),\s*\("negated",\s*negated_text,\s*None\),\s*\("random_masked",\s*masked_text,\s*None\),\s*\]',
        """conditions = [
        ("0-shot",            test_row['cleaned_text'], None),
        ("1-shot",            test_row['cleaned_text'], EXPERT_EXAMPLE),
        ("counterfactual",    test_row['cleaned_text'], COUNTERFACTUAL_EXAMPLE),
        ("negated_hx",        negated_hx_text,          None),
        ("negated_ruled_out", negated_ruled_text,       None),
        ("negated_hedge",     negated_hedge_text,       None),
        ("random_masked",     masked_text,              None),
    ]"""
    ),
    # 4. conditions list B (string tuples)
    (
        r'conditions\s*=\s*\[\s*\("0-shot",\s*test_row\[\'cleaned_text\'\],\s*"0-shot"\),\s*\("1-shot",\s*test_row\[\'cleaned_text\'\],\s*"1-shot"\),\s*\("counterfactual",\s*test_row\[\'cleaned_text\'\],\s*"strong_injection"\),\s*\("negated",\s*negated_text,\s*"0-shot"\),\s*\("random_masked",\s*masked_text,\s*"0-shot"\),\s*\]',
        """conditions = [
        ("0-shot",            test_row['cleaned_text'], "0-shot"),
        ("1-shot",            test_row['cleaned_text'], "1-shot"),
        ("counterfactual",    test_row['cleaned_text'], "strong_injection"),
        ("negated_hx",        negated_hx_text,          "0-shot"),
        ("negated_ruled_out", negated_ruled_text,       "0-shot"),
        ("negated_hedge",     negated_hedge_text,       "0-shot"),
        ("random_masked",     masked_text,              "0-shot"),
    ]"""
    ),
    # 5. ALL_CONDITIONS
    (
        r"ALL_CONDITIONS\s*=\s*\{'0-shot',\s*'1-shot',\s*'counterfactual',\s*'negated',\s*'random_masked'\}",
        "ALL_CONDITIONS = {'0-shot', '1-shot', 'counterfactual', 'negated_hx', 'negated_ruled_out', 'negated_hedge', 'random_masked'}"
    ),
]

# Simple string replacements for print statements
STR_REPLACEMENTS = [
    (' × 5 conditions = ', ' × 7 conditions = '),
    ('x 5 conditions = ', 'x 7 conditions = '),
    ('*5} calls', '*7} calls'),
    ('* 5 calls', '* 7 calls'),
    ('len(df_original)*5', 'len(df_original)*7'),
    ('len(sampled_ids)*5', 'len(sampled_ids)*7'),
    ('len(df_todo)*5', 'len(df_todo)*7'),
    ('len(remaining)*5', 'len(remaining)*7'),
]


def fix_notebook(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    total_changes = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = '\n'.join(cell['source'])
        orig = src

        for pattern, repl in PATTERNS:
            src, n = re.subn(pattern, repl, src, flags=re.DOTALL)
            total_changes += n

        for old, new in STR_REPLACEMENTS:
            if old in src:
                src = src.replace(old, new)
                total_changes += 1

        if src != orig:
            cell['source'] = src.split('\n')

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return total_changes


for fname in ['evennewer_medgemma_inference.ipynb', 'tryadded_qwen3_6_mimic.ipynb']:
    path = DL / fname
    n = fix_notebook(path)
    print(f"Fixed {fname}: {n} replacement(s)")

print("Done.")
