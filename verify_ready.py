"""Verify all notebooks are synced and ready to run."""
import json
from pathlib import Path

DL = Path(r"C:\Users\USER\Downloads\judging_gemma")
errors = []

# ── 1. Check config.py has ALL_CONDITIONS ────────────────────────────────────
config_src = (DL / "judging_gemma" / "config.py").read_text()
if "'negated_hx'" not in config_src:
    errors.append("config.py missing negated_hx in ALL_CONDITIONS")
if "'negated_ruled_out'" not in config_src:
    errors.append("config.py missing negated_ruled_out")
if "'negated_hedge'" not in config_src:
    errors.append("config.py missing negated_hedge")
out_count = config_src.count("DL /")
print(f"[1] config.py: {out_count} path entries found")
if out_count < 40:
    errors.append(f"config.py only has {out_count} paths (expected 42)")

# ── 2. Check judging_utils.py exports ────────────────────────────────────────
utils_src = (DL / "judging_gemma" / "judging_utils.py").read_text()
for fn in ["load_and_fix_gt", "load_clinical_gt", "attach_cleaned_text"]:
    if f"def {fn}" not in utils_src:
        errors.append(f"judging_utils.py missing {fn}")
print("[2] judging_utils.py: all 3 helper functions present")

# ── 3. Check judge notebooks have import cell ────────────────────────────────
judge_nbs = [
    "judge_GLM5.1_llama8b_accuracy.ipynb",
    "judge_GLM5.1_medgemma_accuracy.ipynb",
    "judge_GLM5.1_qwen3_6_accuraci.ipynb",
    "judgeman_dsk_llama8b_accuracy.ipynb",
    "judgeman_dsk_medgemma.ipynb",
    "Judgemandsk_qwen3_6_accuraci.ipynb",
    "judge_GLM5.1_llama8b_reasoning.ipynb",
    "judge_GLM5.1_meggemma_reasoning.ipynb",
    "judge_GLM5.1_qwen3_6_reasoning.ipynb",
    "judgeman_dsk-llama8b_reasoning.ipynb",
    "judgemanmedgemma_reasoning.ipynb",
]
for nb_name in judge_nbs:
    nb = json.loads((DL / nb_name).read_text(encoding='utf-8'))
    src0 = '\n'.join(nb['cells'][0]['source'])
    if 'judging_gemma.config' not in src0:
        errors.append(f"{nb_name}: missing config import")
print(f"[3] Judge notebooks: {len(judge_nbs)} checked, all have import cell")

# ── 4. Check inference notebooks ─────────────────────────────────────────────
inf_nbs = [
    "new_llama_dsk_8b_inference (2).ipynb",
    "evennewer_medgemma_inference.ipynb",
    "tryadded_qwen3_6_mimic.ipynb",
]
for nb_name in inf_nbs:
    nb = json.loads((DL / nb_name).read_text(encoding='utf-8'))
    src = '\n'.join('\n'.join(c['source']) for c in nb['cells'])
    if "neweraugmented_mimic_admission_only.csv" not in src:
        errors.append(f"{nb_name}: missing admission_only.csv path")
    if '"negated_hx"' not in src:
        errors.append(f"{nb_name}: missing negated_hx condition")
    if '"negated_ruled_out"' not in src:
        errors.append(f"{nb_name}: missing negated_ruled_out condition")
    if '"negated_hedge"' not in src:
        errors.append(f"{nb_name}: missing negated_hedge condition")
    cond_count = src.count('"0-shot"') + src.count('"1-shot"') + src.count('"counterfactual"') + src.count('"negated_hx"')
    print(f"[4] {nb_name}: 7-condition entries found")

# ── 5. Check visualization ───────────────────────────────────────────────────
viz = json.loads((DL / "visualize_judge_results.ipynb").read_text(encoding='utf-8'))
viz_src = '\n'.join('\n'.join(c['source']) for c in viz['cells'])
if "negated_hx" not in viz_src:
    errors.append("visualize_judge_results.ipynb: missing negated_hx")
if "d1_icd_accuracy" not in viz_src:
    errors.append("visualize_judge_results.ipynb: missing dual-scoring fallback")
print("[5] visualize_judge_results.ipynb: 7-condition + dual-scoring present")

# ── Report ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if errors:
    print(f"FAIL: {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL CHECKS PASSED — ready to run!")
print("="*50)
