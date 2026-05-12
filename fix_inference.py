"""Fix inference notebooks with proper JSON cell-level editing."""
import json, re
from pathlib import Path

DL = Path(r"C:\Users\USER\Downloads\judging_gemma")

OLD_DATA_PATTERNS = [
    r'DATA_PATH\s*=\s*"/content/drive/MyDrive/neweraugmented_mimic_noleak.csv"',
    r'DATA_PATH\s*=\s*"/content/drive/MyDrive/neweraugmented_mimic.csv"',
]
NEW_DATA = 'DATA_PATH = "/content/drive/MyDrive/neweraugmented_mimic_admission_only.csv"'

def fix_cell(src_lines):
    """Return fixed lines or None if no changes."""
    src = '\n'.join(src_lines)
    changed = False

    # Fix DATA_PATH
    for pat in OLD_DATA_PATTERNS:
        if re.search(pat, src):
            src = re.sub(pat, NEW_DATA, src)
            changed = True

    # Fix negated lookup → 3 lookups
    if "augmentation_type' == 'negated'" in src:
        src = src.replace(
            "negated_text = df[\n        (df['HADM_ID'] == hadm_id) &\n        (df['augmentation_type'] == 'negated')\n    ]['cleaned_text'].values[0]",
            "negated_hx_text = df[\n        (df['HADM_ID'] == hadm_id) &\n        (df['augmentation_type'] == 'negated_hx')\n    ]['cleaned_text'].values[0]\n\n    negated_ruled_text = df[\n        (df['HADM_ID'] == hadm_id) &\n        (df['augmentation_type'] == 'negated_ruled_out')\n    ]['cleaned_text'].values[0]\n\n    negated_hedge_text = df[\n        (df['HADM_ID'] == hadm_id) &\n        (df['augmentation_type'] == 'negated_hedge')\n    ]['cleaned_text'].values[0]"
        )
        changed = True

    # Fix condition lists A (None tuples)
    old_a = '''    conditions = [
        ("0-shot",         test_row['cleaned_text'], None),
        ("1-shot",         test_row['cleaned_text'], EXPERT_EXAMPLE),
        ("counterfactual", test_row['cleaned_text'], COUNTERFACTUAL_EXAMPLE),
        ("negated",        negated_text,             None),
        ("random_masked",  masked_text,              None),
    ]'''
    new_a = '''    conditions = [
        ("0-shot",            test_row['cleaned_text'], None),
        ("1-shot",            test_row['cleaned_text'], EXPERT_EXAMPLE),
        ("counterfactual",    test_row['cleaned_text'], COUNTERFACTUAL_EXAMPLE),
        ("negated_hx",        negated_hx_text,          None),
        ("negated_ruled_out", negated_ruled_text,       None),
        ("negated_hedge",     negated_hedge_text,       None),
        ("random_masked",     masked_text,              None),
    ]'''
    if old_a in src:
        src = src.replace(old_a, new_a)
        changed = True

    # Fix condition lists B (string tuples)
    old_b = '''    conditions = [
        ("0-shot",         test_row['cleaned_text'], "0-shot"),
        ("1-shot",         test_row['cleaned_text'], "1-shot"),
        ("counterfactual", test_row['cleaned_text'], "strong_injection"),
        ("negated",        negated_text,             "0-shot"),
        ("random_masked",  masked_text,              "0-shot"),
    ]'''
    new_b = '''    conditions = [
        ("0-shot",            test_row['cleaned_text'], "0-shot"),
        ("1-shot",            test_row['cleaned_text'], "1-shot"),
        ("counterfactual",    test_row['cleaned_text'], "strong_injection"),
        ("negated_hx",        negated_hx_text,          "0-shot"),
        ("negated_ruled_out", negated_ruled_text,       "0-shot"),
        ("negated_hedge",     negated_hedge_text,       "0-shot"),
        ("random_masked",     masked_text,              "0-shot"),
    ]'''
    if old_b in src:
        src = src.replace(old_b, new_b)
        changed = True

    # Fix ALL_CONDITIONS
    src = src.replace(
        "ALL_CONDITIONS = {'0-shot', '1-shot', 'counterfactual', 'negated', 'random_masked'}",
        "ALL_CONDITIONS = {'0-shot', '1-shot', 'counterfactual', 'negated_hx', 'negated_ruled_out', 'negated_hedge', 'random_masked'}"
    )

    # Fix print statements
    src = src.replace(' × 5 conditions = ', ' × 7 conditions = ')
    src = src.replace('x 5 conditions = ', 'x 7 conditions = ')
    src = src.replace('*5} calls', '*7} calls')
    src = src.replace('* 5 calls', '* 7 calls')
    src = src.replace('len(df_original)*5', 'len(df_original)*7')
    src = src.replace('len(sampled_ids)*5', 'len(sampled_ids)*7')
    src = src.replace('len(df_todo)*5', 'len(df_todo)*7')
    src = src.replace('len(remaining)*5', 'len(remaining)*7')

    if not changed:
        return None
    return src.split('\n')


for fname in ['evennewer_medgemma_inference.ipynb', 'tryadded_qwen3_6_mimic.ipynb']:
    path = DL / fname
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        result = fix_cell(cell['source'])
        if result is not None:
            cell['source'] = result
            changes += 1

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Fixed {fname}: {changes} cell(s) modified")

print("Done.")
