"""Patch all 3 inference notebooks from 5-condition to 7-condition schema.
Also updates DATA_PATH to use neweraugmented_mimic_admission_only.csv."""
import json, re
from pathlib import Path

DL = Path(r"C:\Users\USER\Downloads\judging_gemma")

# ── Old patterns ─────────────────────────────────────────────────────────────
OLD_DATA_PATH = [
    r'DATA_PATH\s*=\s*"/content/drive/MyDrive/neweraugmented_mimic_noleak.csv"',
    r'DATA_PATH\s*=\s*"/content/drive/MyDrive/neweraugmented_mimic.csv"',
]
NEW_DATA_PATH = 'DATA_PATH = "/content/drive/MyDrive/neweraugmented_mimic_admission_only.csv"'

OLD_NEGATED_LOOKUP = """    negated_text = df[
        (df['HADM_ID'] == hadm_id) &
        (df['augmentation_type'] == 'negated')
    ]['cleaned_text'].values[0]"""

NEW_NEGATED_LOOKUP = """    negated_hx_text = df[
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

OLD_CONDITIONS_A = """    conditions = [
        ("0-shot",         test_row['cleaned_text'], None),
        ("1-shot",         test_row['cleaned_text'], EXPERT_EXAMPLE),
        ("counterfactual", test_row['cleaned_text'], COUNTERFACTUAL_EXAMPLE),
        ("negated",        negated_text,             None),
        ("random_masked",  masked_text,              None),
    ]"""

OLD_CONDITIONS_B = """    conditions = [
        ("0-shot",         test_row['cleaned_text'], "0-shot"),
        ("1-shot",         test_row['cleaned_text'], "1-shot"),
        ("counterfactual", test_row['cleaned_text'], "strong_injection"),
        ("negated",        negated_text,             "0-shot"),
        ("random_masked",  masked_text,              "0-shot"),
    ]"""

NEW_CONDITIONS_A = """    conditions = [
        ("0-shot",            test_row['cleaned_text'], None),
        ("1-shot",            test_row['cleaned_text'], EXPERT_EXAMPLE),
        ("counterfactual",    test_row['cleaned_text'], COUNTERFACTUAL_EXAMPLE),
        ("negated_hx",        negated_hx_text,          None),
        ("negated_ruled_out", negated_ruled_text,       None),
        ("negated_hedge",     negated_hedge_text,       None),
        ("random_masked",     masked_text,              None),
    ]"""

NEW_CONDITIONS_B = """    conditions = [
        ("0-shot",            test_row['cleaned_text'], "0-shot"),
        ("1-shot",            test_row['cleaned_text'], "1-shot"),
        ("counterfactual",    test_row['cleaned_text'], "strong_injection"),
        ("negated_hx",        negated_hx_text,          "0-shot"),
        ("negated_ruled_out", negated_ruled_text,       "0-shot"),
        ("negated_hedge",     negated_hedge_text,       "0-shot"),
        ("random_masked",     masked_text,              "0-shot"),
    ]"""

OLD_ALL_CONDITIONS = "ALL_CONDITIONS = {'0-shot', '1-shot', 'counterfactual', 'negated', 'random_masked'}"
NEW_ALL_CONDITIONS = "ALL_CONDITIONS = {'0-shot', '1-shot', 'counterfactual', 'negated_hx', 'negated_ruled_out', 'negated_hedge', 'random_masked'}"


def patch_notebook(nb_path: Path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes = 0
    for cell in nb['cells']:
        src = '\n'.join(cell['source'])

        # Update DATA_PATH
        for old_pat in OLD_DATA_PATH:
            if re.search(old_pat, src):
                src = re.sub(old_pat, NEW_DATA_PATH, src)
                changes += 1

        # Replace negated lookup
        if OLD_NEGATED_LOOKUP in src:
            src = src.replace(OLD_NEGATED_LOOKUP, NEW_NEGATED_LOOKUP)
            changes += 1

        # Replace conditions lists
        if OLD_CONDITIONS_A in src:
            src = src.replace(OLD_CONDITIONS_A, NEW_CONDITIONS_A)
            changes += 1
        if OLD_CONDITIONS_B in src:
            src = src.replace(OLD_CONDITIONS_B, NEW_CONDITIONS_B)
            changes += 1

        # Replace ALL_CONDITIONS
        if OLD_ALL_CONDITIONS in src:
            src = src.replace(OLD_ALL_CONDITIONS, NEW_ALL_CONDITIONS)
            changes += 1

        # Replace print statements
        src = src.replace(' × 5 conditions = ', ' × 7 conditions = ')
        src = src.replace('x 5 conditions = ', 'x 7 conditions = ')
        src = src.replace('*5} calls', '*7} calls')
        src = src.replace('* 5 calls', '* 7 calls')
        src = src.replace('len(df_original)*5', 'len(df_original)*7')
        src = src.replace('len(sampled_ids)*5', 'len(sampled_ids)*7')
        src = src.replace('len(df_todo)*5', 'len(df_todo)*7')
        src = src.replace('len(remaining)*5', 'len(remaining)*7')

        cell['source'] = src.split('\n')

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  Patched {nb_path.name}: {changes} block(s) replaced")


if __name__ == '__main__':
    for fname in ['new_llama_dsk_8b_inference (2).ipynb',
                  'evennewer_medgemma_inference.ipynb',
                  'tryadded_qwen3_6_mimic.ipynb']:
        path = DL / fname
        if path.exists():
            patch_notebook(path)
        else:
            print(f"  SKIP (not found): {fname}")
    print("\nDone.")
