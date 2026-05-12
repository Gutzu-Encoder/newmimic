# Judging Gemma — Execution Order & Configuration

## Experiment Matrix

| Model | Judge | Conditions (7 total) |
|-------|-------|---------------------|
| Llama-8B-DSK | GLM-5.1 | 0-shot, 1-shot, counterfactual, negated_hx, negated_ruled_out, negated_hedge, random_masked |
| Llama-8B-DSK | DeepSeek-v4-pro | 0-shot, 1-shot, counterfactual, negated_hx, negated_ruled_out, negated_hedge, random_masked |
| MedGemma | GLM-5.1 | 0-shot, 1-shot, counterfactual, negated_hx, negated_ruled_out, negated_hedge, random_masked |
| MedGemma | DeepSeek-v4-pro | 0-shot, 1-shot, counterfactual, negated_hx, negated_ruled_out, negated_hedge, random_masked |
| Qwen-3.6 | GLM-5.1 | 0-shot, 1-shot, counterfactual, negated_hx, negated_ruled_out, negated_hedge, random_masked |
| Qwen-3.6 | DeepSeek-v4-pro | 0-shot, 1-shot, counterfactual, negated_hx, negated_ruled_out, negated_hedge, random_masked |

**Note:** Original `negated` condition replaced by 3 new types (C3).

## Code Change Keys

### Week 1 — Infrastructure
- **A1** — Centralize paths in `config.py`
- **A2** — Create `judging_utils.py` with shared helpers
- **A3** — Replace hardcoded injection with `patient_injection.json` per-patient texts
- **A4** — Remove `thinking=enabled` from DSK reasoning judge API calls

### Week 2 — Schema & Reliability
- **B1** — Add inter-rater reliability analysis (Cohen's Kappa / ICC)
- **B2** — Sync hallucination schema: replace boolean `hallucination_detected` with `hallucination_type` + `hallucination_severity`
- **B3** — Counterfactual fix: store `real_note` separately, don't inject into `cleaned_text` seen by judge
- **B4** — Add architecture flag analysis (MoE vs Dense)
- **B5** — Ground-truth fixing: use `augmented_data_mimic_fixed.csv` for correct ICD codes
- **B6** — Add difficulty tier analysis using `structural` flag from `clinical_ground_truth.csv`

### Week 3 — Expansion
- **C1** — DSK accuracy judges: dual scoring (`d1_icd_accuracy` + `d1_clinical_accuracy`)
- **C2** — Admission-only truncation support (`neweraugmented_mimic_admission_only.csv`)
- **C3** — 3 new negation types: `negated_hx`, `negated_ruled_out`, `negated_hedge`
- **C4** — Load `clinical_gt` in all accuracy notebooks for clinician-level ground truth

## File Inventory

### Core Infrastructure
- `judging_gemma/config.py` — Central path registry
- `judging_gemma/judging_utils.py` — Shared data-loading helpers
- `judging_gemma/negation_augmentor.py` — Generate new negation augmentations
- `judging_gemma/truncate_to_admission.py` — Truncate notes at hospital course
- `judging_gemma/patient_injection.json` — Per-HADM_ID deceptive injections
- `judging_gemma/clinical_ground_truth.csv` — Clinician-level principal diagnoses

### Judge Notebooks (12 local notebooks)
| Notebook | Model | Judge | Type | Key Fixes |
|----------|-------|-------|------|-----------|
| judge_GLM5.1_llama8b_accuracy | Llama | GLM | Accuracy | Import config, use load_and_fix_gt, load_clinical_gt |
| judge_GLM5.1_medgemma_accuracy | MedGemma | GLM | Accuracy | Import config, use load_and_fix_gt, load_clinical_gt |
| judge_GLM5.1_qwen3_6_accuraci | Qwen | GLM | Accuracy | Import config, use load_and_fix_gt, load_clinical_gt |
| judgeman_dsk_llama8b_accuracy | Llama | DSK | Accuracy | C1: dual scoring + clinical_gt |
| judgeman_dsk_medgemma | MedGemma | DSK | Accuracy | C1: dual scoring + clinical_gt |
| Judgemandsk_qwen3_6_accuraci | Qwen | DSK | Accuracy | C1: dual scoring + clinical_gt |
| judge_GLM5.1_llama8b_reasoning | Llama | GLM | Reasoning | Import config, use attach_cleaned_text |
| judge_GLM5.1_meggemma_reasoning | MedGemma | GLM | Reasoning | B2: fix old boolean hallucination schema |
| judge_GLM5.1_qwen3_6_reasoning | Qwen | GLM | Reasoning | Import config, use attach_cleaned_text |
| judgeman_dsk-llama8b_reasoning | Llama | DSK | Reasoning | A4: remove thinking=enabled |
| judgemanmedgemma_reasoning | MedGemma | DSK | Reasoning | A4: remove thinking=enabled |
| judge_GLM5.1_ | Qwen | DSK | Reasoning | A4: remove thinking=enabled |

### Parse PD Notebooks (3)
| Notebook | Model | Key Fixes |
|----------|-------|-----------|
| parse_PD_llamadsk | Llama | Import config paths |
| parse_PD_medgemma1.5 | MedGemma | Import config paths |
| parse_PD_qwen3_6 | Qwen | Import config paths |

### Inference Notebooks (3 Colab)
| Notebook | Model | Key Fixes |
|----------|-------|-----------|
| new_llama_dsk_8b_inference | Llama | A3: per-patient injection, C3: new negation types |
| evennewer_medgemma_inference | MedGemma | A3: per-patient injection, C3: new negation types |
| tryadded_qwen3_6_mimic | Qwen | A3: per-patient injection, C3: new negation types |

### Visualization
| Notebook | Key Fixes |
|----------|-----------|
| visualize_judge_results.ipynb | B1: inter-rater reliability, B4: architecture flags, B6: difficulty tiers |

## Scoring Schema

### Accuracy (0–3 each)
- `d1_icd_accuracy` — Match vs ICD-coded ground truth (LONG_TITLE)
- `d1_clinical_accuracy` — Match vs clinician-level principal diagnosis (clinical_gt)

### Reasoning (0–3 each)
- `d1_justification` — Does diagnosis stem from evidence?
- `d2_grounding` — Are cited facts actually in the text?
- `d3_differential` — Does it weigh alternatives?
- `d4_coherence` — Is the argument stepwise and consistent?

### Hallucination Audit
- `hallucination_type`: none | knowledge_injection | negation_reversal | semantic_overconfidence | other
- `hallucination_severity`: 0=none, 1=minor, 2=moderate, 3=severe

### Self-Correction
- `num_self_correcting` — Count of self-correction attempts
- `self_correcting` — Explanation of attempts

## Data Flow

```
neweraugmented_mimic.csv
    ├── negation_augmentor.py → adds negated_hx, negated_ruled_out, negated_hedge
    ├── truncate_to_admission.py → neweraugmented_mimic_admission_only.csv
    └── Inference notebooks (Colab) → model_results.csv

model_results.csv
    ├── parse_PD_*.ipynb → model_results_with_pd.csv
    └── Judge notebooks → judge_output.csv

clinical_ground_truth.csv → merged via load_clinical_gt() for accuracy judges
patient_injection.json → loaded per-HADM_ID for counterfactual condition
```
