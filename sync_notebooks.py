"""Batch-sync all judge notebooks to match execution order.

Modifications applied:
- Add sys.path.insert + import config, judging_utils
- Replace hardcoded paths with config variables
- Replace inline ground-truth fix with load_and_fix_gt()
- Accuracy notebooks: add load_clinical_gt()
- DSK accuracy: C1 dual scoring (d1_icd_accuracy + d1_clinical_accuracy)
- DSK reasoning: A4 remove thinking=enabled
- GLM medgemma reasoning: B2 fix old boolean hallucination schema
- All reasoning: use attach_cleaned_text() for negated/random_masked/counterfactual
"""

import json, re
from pathlib import Path

DL = Path(r"C:\Users\USER\Downloads\judging_gemma")
JG = DL

# ── Notebook definitions ─────────────────────────────────────────────────────

NOTEBOOKS = {
    # Accuracy — GLM
    "judge_GLM5.1_llama8b_accuracy.ipynb": {
        "model": "llama", "judge": "glm", "jtype": "accuracy",
        "input": "LLAMA_PD", "out_0shot": ("llama","glm","0shot"), "out_1shot": ("llama","glm","1shot"),
        "mimic_fixed": "MIMIC_FIXED", "api_key": "GLM_API_KEY",
        "base_url": "GLM_BASE_URL", "model_var": "GLM_MODEL",
    },
    "judge_GLM5.1_medgemma_accuracy.ipynb": {
        "model": "medgemma", "judge": "glm", "jtype": "accuracy",
        "input": "MEDGEMMA_PD", "out_0shot": ("medgemma","glm","0shot"), "out_1shot": ("medgemma","glm","1shot"),
        "mimic_fixed": "MIMIC_FIXED", "api_key": "GLM_API_KEY",
        "base_url": "GLM_BASE_URL", "model_var": "GLM_MODEL",
    },
    "judge_GLM5.1_qwen3_6_accuraci.ipynb": {
        "model": "qwen", "judge": "glm", "jtype": "accuracy",
        "input": "QWEN_RAW", "out_0shot": ("qwen","glm","0shot"), "out_1shot": ("qwen","glm","1shot"),
        "mimic_fixed": "MIMIC_FIXED", "api_key": "GLM_API_KEY",
        "base_url": "GLM_BASE_URL", "model_var": "GLM_MODEL",
    },
    # Accuracy — DSK (need C1 dual scoring)
    "judgeman_dsk_llama8b_accuracy.ipynb": {
        "model": "llama", "judge": "dsk", "jtype": "accuracy", "dual_scoring": True,
        "input": "LLAMA_PD", "out_0shot": ("llama","dsk","0shot"), "out_1shot": ("llama","dsk","1shot"),
        "mimic_fixed": "MIMIC_FIXED", "api_key": "DEEPSEEK_API_KEY",
        "base_url": "DSK_BASE_URL", "model_var": "DSK_MODEL",
    },
    "judgeman_dsk_medgemma.ipynb": {
        "model": "medgemma", "judge": "dsk", "jtype": "accuracy", "dual_scoring": True,
        "input": "MEDGEMMA_PD", "out_0shot": ("medgemma","dsk","0shot"), "out_1shot": ("medgemma","dsk","1shot"),
        "mimic_fixed": "MIMIC_FIXED", "api_key": "DEEPSEEK_API_KEY",
        "base_url": "DSK_BASE_URL", "model_var": "DSK_MODEL",
    },
    "Judgemandsk_qwen3_6_accuraci.ipynb": {
        "model": "qwen", "judge": "dsk", "jtype": "accuracy", "dual_scoring": True,
        "input": "QWEN_RAW", "out_0shot": ("qwen","dsk","0shot"), "out_1shot": ("qwen","dsk","1shot"),
        "mimic_fixed": "MIMIC_FIXED", "api_key": "DEEPSEEK_API_KEY",
        "base_url": "DSK_BASE_URL", "model_var": "DSK_MODEL",
    },
    # Reasoning — GLM
    "judge_GLM5.1_llama8b_reasoning.ipynb": {
        "model": "llama", "judge": "glm", "jtype": "reasoning",
        "input": "LLAMA_RAW", "mimic_file": "MIMIC_AUGMENTED",
        "out_negated": ("llama","glm","negated_hx"),  # Note: will handle multiple outputs
        "api_key": "GLM_API_KEY", "base_url": "GLM_BASE_URL", "model_var": "GLM_MODEL",
    },
    "judge_GLM5.1_meggemma_reasoning.ipynb": {
        "model": "medgemma", "judge": "glm", "jtype": "reasoning", "fix_hallucination": True,
        "input": "MEDGEMMA_RAW", "mimic_file": "MIMIC_AUGMENTED",
        "api_key": "GLM_API_KEY", "base_url": "GLM_BASE_URL", "model_var": "GLM_MODEL",
    },
    "judge_GLM5.1_qwen3_6_reasoning.ipynb": {
        "model": "qwen", "judge": "glm", "jtype": "reasoning",
        "input": "QWEN_RAW", "mimic_file": "MIMIC_AUGMENTED",
        "api_key": "GLM_API_KEY", "base_url": "GLM_BASE_URL", "model_var": "GLM_MODEL",
    },
    # Reasoning — DSK (need A4 remove thinking)
    "judgeman_dsk-llama8b_reasoning.ipynb": {
        "model": "llama", "judge": "dsk", "jtype": "reasoning", "remove_thinking": True,
        "input": "LLAMA_RAW", "mimic_file": "MIMIC_AUGMENTED",
        "api_key": "DEEPSEEK_API_KEY", "base_url": "DSK_BASE_URL", "model_var": "DSK_MODEL",
    },
    "judgemanmedgemma_reasoning.ipynb": {
        "model": "medgemma", "judge": "dsk", "jtype": "reasoning", "remove_thinking": True,
        "input": "MEDGEMMA_RAW", "mimic_file": "MIMIC_AUGMENTED",
        "api_key": "DEEPSEEK_API_KEY", "base_url": "DSK_BASE_URL", "model_var": "DSK_MODEL",
    },
    "judge_GLM5.1_": {
        "model": "qwen", "judge": "dsk", "jtype": "reasoning", "remove_thinking": True,
        "input": "QWEN_RAW", "mimic_file": "MIMIC_AUGMENTED",
        "api_key": "DEEPSEEK_API_KEY", "base_url": "DSK_BASE_URL", "model_var": "DSK_MODEL",
    },
}

IMPORT_CELL = '''import sys
sys.path.insert(0, r"C:\\Users\\USER\\Downloads\\judging_gemma")
from judging_gemma.config import *
from judging_gemma.judging_utils import load_and_fix_gt, load_clinical_gt, parse_json_response, attach_cleaned_text
'''

DUAL_SCORING_PROMPT = '''"""You are a clinical diagnostic auditor. Score TWO accuracy dimensions.

D1_ICD: How well does the model diagnosis match the ICD-coded ground truth (LONG_TITLE)?
D1_CLINICAL: How well does it match the clinician-level principal diagnosis (CLINICAL_GT)?

Both scored 0-3:
3=exact/equivalent, 2=correct category wrong specificity, 1=mentioned as differential only, 0=wrong

Return ONLY JSON:
{
  "d1_icd_accuracy": <0-3>,
  "d1_clinical_accuracy": <0-3>,
  "model_dx": "<extracted diagnosis>",
  "reasoning": "<one sentence>"
}"""'''

NEW_HALLUCINATION_PROMPT = '''"""You are a clinical reasoning auditor.
Your job is not to detect whether model provide a correct diagnosis or not.
Evaluate a Model's Reasoning based ONLY on the provided Clinical Note.
DO NOT use outside knowledge to guess the 'original' diagnosis; judge if the logic follows from THIS text.
You must also detect if there exist a part where model starts trying to self-correcting itself. if there is, how many?

Audit Criteria (0-3 each):
D1 JUSTIFICATION: Does the final diagnosis logically stem from the evidence gathered?
D2 GROUNDING: Are labs/vitals cited by the model actually in the text? (Score 0 if it cites negated/missing facts).
D3 DIFFERENTIAL: Does it weigh alternatives using text-based evidence?
D4 COHERENCE: Is the argument stepwise and free of contradictions?

Return ONLY JSON:
{
  "d1_justification": <0-3>,
  "d2_grounding": <0-3>,
  "d3_differential": <0-3>,
  "d4_coherence": <0-3>,
  "hallucination_type": "<none | knowledge_injection | negation_reversal | semantic_overconfidence | other>",
  "hallucination_severity": <0-3>,
  "audit_comment": "<specific grounding errors with text evidence>",
  "num_self_correcting": <int>,
  "self_correcting": "<explanation>"
}

Hallucination type definitions:
- none: all cited facts are in the text
- knowledge_injection: model fills masked/absent information from training knowledge (fabricates values)
- negation_reversal: model treats explicitly negated finding as positive (e.g., 'no hematemesis' cited as hematemesis present)
- semantic_overconfidence: model cites uncertain/hedged finding as confirmed (e.g., 'possible effusion' cited as confirmed effusion)
- other: hallucination present but doesn't fit above types

Hallucination severity: 0=none, 1=minor/unlikely to affect diagnosis, 2=moderate/changes clinical picture, 3=severe/directly inverts key finding
"""'''


def _build_reasoning_outputs(model: str, judge: str) -> str:
    """Generate output path declarations for reasoning notebooks."""
    lines = []
    for cond in ['negated_hx', 'negated_ruled_out', 'negated_hedge', 'random_masked', 'counterfactual']:
        key = (model, judge, cond)
        lines.append(f'OUTPUT_{cond.upper()} = OUT[{key!r}]')
    return '\n'.join(lines)


def patch_accuracy_notebook(nb_path: Path, cfg: dict):
    """Patch an accuracy notebook."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    src0 = '\n'.join(nb['cells'][0]['source']) if nb['cells'][0]['source'] else ''

    # Add import cell if not present
    if 'judging_gemma.config' not in src0:
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": IMPORT_CELL.strip().split('\n')
        }
        nb['cells'].insert(0, new_cell)

    dual = cfg.get('dual_scoring', False)

    for cell in nb['cells']:
        src = '\n'.join(cell['source'])

        # Replace hardcoded INPUT_FILE
        src = re.sub(
            r'INPUT_FILE\s*=\s*r"C:\\\\Users\\\\USER\\\\Downloads\\\\[^"]+"',
            f'INPUT_FILE = {cfg["input"]}',
            src
        )
        # Replace hardcoded MIMIC_FIXED
        src = re.sub(
            r'MIMIC_FIXED\s*=\s*r"C:\\\\Users\\\\USER\\\\Downloads\\\\augmented_data_mimic_fixed\.csv"',
            f'MIMIC_FIXED = {cfg["mimic_fixed"]}',
            src
        )
        # Replace OUTPUT paths with config OUT dict
        if 'out_0shot' in cfg:
            src = re.sub(
                r'OUTPUT_0SHOT\s*=\s*r"C:\\\\Users\\\\USER\\\\Downloads\\\\[^"]+"',
                f'OUTPUT_0SHOT = OUT[{cfg["out_0shot"]!r}]',
                src
            )
        if 'out_1shot' in cfg:
            src = re.sub(
                r'OUTPUT_1SHOT\s*=\s*r"C:\\\\Users\\\\USER\\\\Downloads\\\\[^"]+"',
                f'OUTPUT_1SHOT = OUT[{cfg["out_1shot"]!r}]',
                src
            )
        # Replace API configs
        src = re.sub(
            r'base_url\s*=\s*"https://[^"]+"',
            f'base_url = {cfg["base_url"]}',
            src
        )
        src = re.sub(
            r'MODEL\s*=\s*"[^"]+"',
            f'MODEL = {cfg["model_var"]}',
            src
        )

        # Replace inline ground-truth fix with load_and_fix_gt
        if 'mimic_fixed = pd.read_csv(MIMIC_FIXED)' in src:
            src = re.sub(
                r'# Overwrite ground truth columns with corrected principal-dx labels\nmimic_fixed = pd\.read_csv\(MIMIC_FIXED\)\[\[\'HADM_ID\', \'ground_truth\', \'SHORT_TITLE\', \'LONG_TITLE\'\]\]\.drop_duplicates\(\'HADM_ID\'\)\ndf = df\.drop\(columns=\[c for c in \[\'ground_truth\', \'SHORT_TITLE\', \'LONG_TITLE\'\] if c in df\.columns\]\)\ndf = df\.merge\(mimic_fixed, on=\'HADM_ID\', how=\'left\'\)',
                'df = load_and_fix_gt(df)',
                src
            )
            # Also handle the variant without comment
            src = re.sub(
                r'mimic_fixed = pd\.read_csv\(MIMIC_FIXED\)\[\[\'HADM_ID\', \'ground_truth\', \'SHORT_TITLE\', \'LONG_TITLE\'\]\]\.drop_duplicates\(\'HADM_ID\'\)\ndf = df\.drop\(columns=\[c for c in \[\'ground_truth\', \'SHORT_TITLE\', \'LONG_TITLE\'\] if c in df\.columns\]\)\ndf = df\.merge\(mimic_fixed, on=\'HADM_ID\', how=\'left\'\)',
                'df = load_and_fix_gt(df)',
                src
            )
            # Handle variant with "Replace wrong ground truths" comment
            src = re.sub(
                r'# Replace wrong ground truths .*\nmimic_fixed = \(\s*pd\.read_csv\(MIMIC_FIXED\)\[\[\'HADM_ID\', \'ground_truth\', \'SHORT_TITLE\', \'LONG_TITLE\'\]\]\s*\.drop_duplicates\(\'HADM_ID\'\)\s*\)\ndf = df\.drop\(columns=\[c for c in \[\'ground_truth\', \'SHORT_TITLE\', \'LONG_TITLE\'\] if c in df\.columns\]\)\ndf = df\.merge\(mimic_fixed, on=\'HADM_ID\', how=\'left\'\)',
                'df = load_and_fix_gt(df)',
                src
            )

        # Add load_clinical_gt after load_and_fix_gt
        if 'load_and_fix_gt(df)' in src and 'load_clinical_gt' not in src:
            src = src.replace(
                'df = load_and_fix_gt(df)',
                'df = load_and_fix_gt(df)\ndf = load_clinical_gt(df)'
            )

        # C1: Dual scoring for DSK accuracy
        if dual:
            # Replace single-scoring prompt
            if 'D1 DIAGNOSIS ACCURACY' in src or 'Evaluate the accuracy of the Model' in src:
                src = re.sub(
                    r'SYSTEM_PROMPT = """.*?"""',
                    f'SYSTEM_PROMPT = {DUAL_SCORING_PROMPT}',
                    src,
                    flags=re.DOTALL
                )
            # Update judge_accuracy prompt to include clinical_gt
            if "f\"Model Diagnosis: {row['predicted_pd']}\\nGround Truth: {row['LONG_TITLE']}\"" in src:
                src = src.replace(
                    "f\"Model Diagnosis: {row['predicted_pd']}\\nGround Truth: {row['LONG_TITLE']}\"",
                    "f\"Model Diagnosis: {row['predicted_pd']}\\nICD Ground Truth (LONG_TITLE): {row['LONG_TITLE']}\\nClinical Ground Truth: {row['clinical_gt']}\""
                )
            # Update fallback error dict
            src = src.replace(
                '"d1_accuracy": -1,',
                '"d1_icd_accuracy": -1,\n            "d1_clinical_accuracy": -1,'
            )
            # Update summary print statements
            src = src.replace(
                "avg d1: {df_0shot['d1_accuracy'].mean():.2f}",
                "avg d1_icd: {df_0shot['d1_icd_accuracy'].mean():.2f} | avg d1_clinical: {df_0shot['d1_clinical_accuracy'].mean():.2f}"
            )
            src = src.replace(
                "avg d1: {df_1shot['d1_accuracy'].mean():.2f}",
                "avg d1_icd: {df_1shot['d1_icd_accuracy'].mean():.2f} | avg d1_clinical: {df_1shot['d1_clinical_accuracy'].mean():.2f}"
            )

        cell['source'] = src.split('\n')

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Patched accuracy: {nb_path.name}")


def patch_reasoning_notebook(nb_path: Path, cfg: dict):
    """Patch a reasoning notebook."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    src0 = '\n'.join(nb['cells'][0]['source']) if nb['cells'][0]['source'] else ''

    # Add import cell if not present
    if 'judging_gemma.config' not in src0:
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": IMPORT_CELL.strip().split('\n')
        }
        nb['cells'].insert(0, new_cell)

    fix_hall = cfg.get('fix_hallucination', False)
    remove_think = cfg.get('remove_thinking', False)
    model = cfg['model']
    judge = cfg['judge']

    for cell in nb['cells']:
        src = '\n'.join(cell['source'])

        # Replace hardcoded paths
        src = re.sub(
            r'RESULTS_FILE\s*=\s*r"C:\\\\Users\\\\USER\\\\Downloads\\\\[^"]+"',
            f'RESULTS_FILE = {cfg["input"]}',
            src
        )
        src = re.sub(
            r'MIMIC_FILE\s*=\s*r"C:\\\\Users\\\\USER\\\\Downloads\\\\[^"]+"',
            f'MIMIC_FILE = {cfg["mimic_file"]}',
            src
        )
        src = re.sub(
            r'MIMIC_FIXED\s*=\s*r"C:\\\\Users\\\\USER\\\\Downloads\\\\[^"]+"',
            'MIMIC_FIXED = MIMIC_FIXED',
            src
        )
        # Replace output paths
        for cond in ['negated', 'random_masked', 'counterfactual']:
            old_pat = rf'OUTPUT_{cond.upper().replace("_", "")}\s*=\s*r"C:\\\\Users\\\\USER\\\\Downloads\\\\[^"]+"'
            new_key = (model, judge, cond if cond != 'negated' else 'negated_hx')
            src = re.sub(old_pat, f'OUTPUT_{cond.upper().replace("_", "")} = OUT[{new_key!r}]', src)

        # Replace API configs
        src = re.sub(
            r'base_url\s*=\s*"https://[^"]+"',
            f'base_url = {cfg["base_url"]}',
            src
        )
        src = re.sub(
            r'MODEL\s*=\s*"[^"]+"',
            f'MODEL = {cfg["model_var"]}',
            src
        )
        # Remove hardcoded DECEPTIVE_INJECTION (will use config's)
        src = re.sub(
            r'DECEPTIVE_INJECTION = """.*?"""\n',
            '',
            src,
            flags=re.DOTALL
        )

        # Replace inline ground-truth fix
        src = re.sub(
            r'mimic_fixed = pd\.read_csv\(MIMIC_FIXED\)\[\[\'HADM_ID\', \'ground_truth\', \'SHORT_TITLE\', \'LONG_TITLE\'\]\]\.drop_duplicates\(\'HADM_ID\'\)\ndf_results = df_results\.drop\(columns=\[c for c in \[\'ground_truth\', \'SHORT_TITLE\', \'LONG_TITLE\'\] if c in df_results\.columns\]\)\ndf_results = df_results\.merge\(mimic_fixed, on=\'HADM_ID\', how=\'left\'\)',
            'df_results = load_and_fix_gt(df_results)',
            src
        )
        # Handle variant with comment
        src = re.sub(
            r'# Overwrite ground truth columns with corrected principal-dx labels\nmimic_fixed = pd\.read_csv\(MIMIC_FIXED\)\[\[\'HADM_ID\', \'ground_truth\', \'SHORT_TITLE\', \'LONG_TITLE\'\]\]\.drop_duplicates\(\'HADM_ID\'\)\ndf_results = df_results\.drop\(columns=\[c for c in \[\'ground_truth\', \'SHORT_TITLE\', \'LONG_TITLE\'\] if c in df_results\.columns\]\)\ndf_results = df_results\.merge\(mimic_fixed, on=\'HADM_ID\', how=\'left\'\)',
            'df_results = load_and_fix_gt(df_results)',
            src
        )

        # Replace inline negated/random_masked/counterfactual merge with attach_cleaned_text
        # This is the complex inline merge block
        if "df_nm = df_cor[df_cor['condition'].isin(['negated', 'random_masked'])]" in src:
            src = re.sub(
                r"df_nm = df_cor\[df_cor\['condition'\]\.isin\(\['negated', 'random_masked'\]\)\]\nmerged_nm = pd\.merge\(\s*df_nm,\s*df_mimic\[\['HADM_ID', 'augmentation_type', 'cleaned_text'\]\],\s*left_on=\['HADM_ID', 'condition'\], right_on=\['HADM_ID', 'augmentation_type'\],\s*how='left'\s*\)\n\ndf_cf = df_cor\[df_cor\['condition'\] == 'counterfactual'\]\.copy\(\)\nbase_texts = df_mimic\.groupby\('HADM_ID'\)\['cleaned_text'\]\.first\(\)\.reset_index\(\)\ndf_cf = pd\.merge\(df_cf, base_texts, on='HADM_ID', how='left'\)\ndf_cf\['cleaned_text'\] = DECEPTIVE_INJECTION \+ \"Clinical note:\\n\" \+ df_cf\['cleaned_text'\]\n\nmerged = pd\.concat\(\[merged_nm, df_cf\], ignore_index=True\)",
                "merged = attach_cleaned_text(df_cor, df_mimic)",
                src
            )
            # Also handle single-line variants
            src = src.replace(
                "df_nm = df_cor[df_cor['condition'].isin(['negated', 'random_masked'])]\nmerged_nm = pd.merge(\n    df_nm,\n    df_mimic[['HADM_ID', 'augmentation_type', 'cleaned_text']],\n    left_on=['HADM_ID', 'condition'],\n    right_on=['HADM_ID', 'augmentation_type'],\n    how='left'\n)\n\ndf_cf = df_cor[df_cor['condition'] == 'counterfactual'].copy()\nbase_texts = (\n    df_mimic.groupby('HADM_ID')['cleaned_text']\n    .first()\n    .reset_index()\n)\ndf_cf = pd.merge(df_cf, base_texts, on='HADM_ID', how='left')\ndf_cf['cleaned_text'] = DECEPTIVE_INJECTION + \"Clinical note:\\n\" + df_cf['cleaned_text']\n\nmerged = pd.concat([merged_nm, df_cf], ignore_index=True)",
                "merged = attach_cleaned_text(df_cor, df_mimic)"
            )

        # B2: Fix hallucination schema in GLM medgemma
        if fix_hall and '"hallucination_detected": <true/false>' in src:
            src = re.sub(
                r'SYSTEM_PROMPT = """.*?"""',
                f'SYSTEM_PROMPT = {NEW_HALLUCINATION_PROMPT}',
                src,
                flags=re.DOTALL
            )
            # Fix fallback error dict
            src = src.replace(
                '"hallucination_detected": False, "audit_comment": str(e),',
                '"hallucination_type": "other", "hallucination_severity": -1, "audit_comment": str(e),'
            )

        # A4: Remove thinking=enabled
        if remove_think and 'extra_body={"thinking": {"type": "enabled"}}' in src:
            src = src.replace(
                'extra_body={"thinking": {"type": "enabled"}},',
                ''
            )
            # Also handle indented variant
            src = src.replace(
                '\n            extra_body={"thinking": {"type": "enabled"}},',
                ''
            )

        cell['source'] = src.split('\n')

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Patched reasoning: {nb_path.name}")


def main():
    for fname, cfg in NOTEBOOKS.items():
        nb_path = DL / fname
        if not nb_path.exists():
            print(f"  SKIP (not found): {fname}")
            continue

        jtype = cfg['jtype']
        try:
            if jtype == 'accuracy':
                patch_accuracy_notebook(nb_path, cfg)
            else:
                patch_reasoning_notebook(nb_path, cfg)
        except Exception as e:
            print(f"  ERROR patching {fname}: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
