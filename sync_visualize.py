"""Patch visualize_judge_results.ipynb for 7-condition schema + dual scoring."""
import json
from pathlib import Path

DL = Path(r"C:\Users\USER\Downloads\judging_gemma")
nb_path = DL / "visualize_judge_results.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = '\n'.join(cell['source'])

    # ── 1. Expand FILE_REGISTRY: negated → 3 types ──────────────────────────
    # GLM reasoning
    src = src.replace(
        '(DOWNLOADS / "llama8b_glmjudge_negated.csv",        "Llama-8B-DSK", "GLM-5.1", "negated",      "reasoning"),',
        '(DOWNLOADS / "llama8b_glmjudge_negated_hx.csv",      "Llama-8B-DSK", "GLM-5.1", "negated_hx",      "reasoning"),\n    (DOWNLOADS / "llama8b_glmjudge_negated_ruled_out.csv", "Llama-8B-DSK", "GLM-5.1", "negated_ruled_out","reasoning"),\n    (DOWNLOADS / "llama8b_glmjudge_negated_hedge.csv",    "Llama-8B-DSK", "GLM-5.1", "negated_hedge",   "reasoning"),'
    )
    src = src.replace(
        '(DOWNLOADS / "medgemma_glmjudge_negated.csv",       "MedGemma",     "GLM-5.1", "negated",      "reasoning"),',
        '(DOWNLOADS / "medgemma_glmjudge_negated_hx.csv",     "MedGemma",     "GLM-5.1", "negated_hx",      "reasoning"),\n    (DOWNLOADS / "medgemma_glmjudge_negated_ruled_out.csv","MedGemma",     "GLM-5.1", "negated_ruled_out","reasoning"),\n    (DOWNLOADS / "medgemma_glmjudge_negated_hedge.csv",   "MedGemma",     "GLM-5.1", "negated_hedge",   "reasoning"),'
    )
    src = src.replace(
        '(DOWNLOADS / "qwen36_glmjudge_negated.csv",         "Qwen-3.6",     "GLM-5.1", "negated",      "reasoning"),',
        '(DOWNLOADS / "qwen36_glmjudge_negated_hx.csv",       "Qwen-3.6",     "GLM-5.1", "negated_hx",      "reasoning"),\n    (DOWNLOADS / "qwen36_glmjudge_negated_ruled_out.csv", "Qwen-3.6",     "GLM-5.1", "negated_ruled_out","reasoning"),\n    (DOWNLOADS / "qwen36_glmjudge_negated_hedge.csv",    "Qwen-3.6",     "GLM-5.1", "negated_hedge",   "reasoning"),'
    )

    # DSK reasoning
    src = src.replace(
        '(DOWNLOADS / "llama8b_judge_negated.csv",           "Llama-8B-DSK", "DeepSeek","negated",      "reasoning"),',
        '(DOWNLOADS / "llama8b_judge_negated_hx.csv",         "Llama-8B-DSK", "DeepSeek","negated_hx",      "reasoning"),\n    (DOWNLOADS / "llama8b_judge_negated_ruled_out.csv",  "Llama-8B-DSK", "DeepSeek","negated_ruled_out","reasoning"),\n    (DOWNLOADS / "llama8b_judge_negated_hedge.csv",     "Llama-8B-DSK", "DeepSeek","negated_hedge",   "reasoning"),'
    )
    src = src.replace(
        '(DOWNLOADS / "medgemma_judge_negated.csv",          "MedGemma",     "DeepSeek","negated",      "reasoning"),',
        '(DOWNLOADS / "medgemma_judge_negated_hx.csv",        "MedGemma",     "DeepSeek","negated_hx",      "reasoning"),\n    (DOWNLOADS / "medgemma_judge_negated_ruled_out.csv", "MedGemma",     "DeepSeek","negated_ruled_out","reasoning"),\n    (DOWNLOADS / "medgemma_judge_negated_hedge.csv",    "MedGemma",     "DeepSeek","negated_hedge",   "reasoning"),'
    )

    # ── 2. Accuracy dual-scoring support ─────────────────────────────────────
    # Replace d1_accuracy check with dual-scoring fallback
    src = src.replace(
        "if not df_acc.empty and 'd1_accuracy' in df_acc.columns:",
        "if not df_acc.empty and ('d1_accuracy' in df_acc.columns or 'd1_icd_accuracy' in df_acc.columns):"
    )
    src = src.replace(
        "df_acc = df_acc[df_acc['d1_accuracy'].between(0, 3)]",
        "score_col = 'd1_icd_accuracy' if 'd1_icd_accuracy' in df_acc.columns else 'd1_accuracy'\n    df_acc = df_acc[df_acc[score_col].between(0, 3)]"
    )
    src = src.replace(
        ".groupby(['model', 'condition', 'judge'])['d1_accuracy']",
        ".groupby(['model', 'condition', 'judge'])[score_col]"
    )
    src = src.replace(
        ".rename(columns={'d1_accuracy': 'mean_d1'})",
        ".rename(columns={score_col: 'mean_d1'})"
    )
    # Also update summary export for accuracy
    src = src.replace(
        "'d1_accuracy_mean': grp['d1_accuracy'].mean(),",
        "'d1_accuracy_mean': grp[score_col].mean() if 'd1_accuracy' in grp.columns else grp['d1_icd_accuracy'].mean(),"
    )
    src = src.replace(
        "'d1_accuracy_std': grp['d1_accuracy'].std(),",
        "'d1_accuracy_std': grp[score_col].std() if 'd1_accuracy' in grp.columns else grp['d1_icd_accuracy'].std(),"
    )

    # ── 3. Reasoning col_order ───────────────────────────────────────────────
    src = src.replace(
        "col_order=['negated', 'random_masked', 'counterfactual']",
        "col_order=['negated_hx', 'negated_ruled_out', 'negated_hedge', 'random_masked', 'counterfactual']"
    )

    # ── 4. Hallucination: boolean → type/severity schema ─────────────────────
    src = src.replace(
        "if not df_reason.empty and 'hallucination_detected' in df_reason.columns:",
        "if not df_reason.empty and 'hallucination_type' in df_reason.columns:"
    )
    src = src.replace(
        "df_reason.groupby(['model', 'condition', 'judge'])['hallucination_detected']",
        "df_reason.groupby(['model', 'condition', 'judge'])['hallucination_type']"
    )
    src = src.replace(
        ".apply(lambda x: x.astype(bool).mean() * 100)",
        ".apply(lambda x: (x != 'none').mean() * 100)"
    )
    src = src.replace(
        ".rename(columns={'hallucination_detected': 'hallucination_pct'})",
        ".rename(columns={'hallucination_type': 'hallucination_pct'})"
    )
    # Summary export hallucination
    src = src.replace(
        "if 'hallucination_detected' in grp.columns:",
        "if 'hallucination_type' in grp.columns:"
    )
    src = src.replace(
        "rec['hallucination_rate_pct'] = grp['hallucination_detected'].astype(bool).mean() * 100",
        "rec['hallucination_rate_pct'] = (grp['hallucination_type'] != 'none').mean() * 100"
    )

    # ── 5. Self-correction condition lists ───────────────────────────────────
    src = src.replace(
        "for ax2, cond in zip(axes, ['negated', 'random_masked', 'counterfactual']):",
        "for ax2, cond in zip(axes, ['negated_hx', 'negated_ruled_out', 'negated_hedge', 'random_masked', 'counterfactual']):"
    )
    # Only keep 3 subplots for histogram, so change to 1 row with shared logic
    src = src.replace(
        "fig2, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)",
        "fig2, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)\n    axes = axes.flatten()"
    )
    src = src.replace(
        "fig2.suptitle('Self-Correction Distribution by Condition', fontsize=14, fontweight='bold')",
        "fig2.suptitle('Self-Correction Distribution by Condition', fontsize=14, fontweight='bold')\n    for ax in axes[5:]:\n        ax.set_visible(False)"
    )

    cell['source'] = src.split('\n')

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Patched {nb_path.name}")
