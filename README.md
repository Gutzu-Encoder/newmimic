# Clinical Principal Diagnosis Evaluation — Project Overview

## What We Are Doing

We test whether large language models (LLMs) can correctly identify the **principal diagnosis** from real de-identified hospital discharge notes (MIMIC-III). We deliberately distort the notes in 6 different ways to probe *where* and *why* models fail. Two separate LLM judges then score the model outputs on accuracy and clinical reasoning quality.

**Models evaluated:** MedGemma-1.5-4B, Llama-8B-DSK-Distill, Qwen-3.6B  
**Judges:** DeepSeek-R1, GLM-5.1  
**Patient cases:** 10 HADM_IDs from MIMIC-III (5 oncology / 5 respiratory)

---

## Pipeline — Data Flow

```
MIMIC-III discharge notes
        │
        ▼
[1] CLEAN & TRUNCATE          truncate_to_admission.py
    Strip post-discharge text,
    keep only admission note
        │
        ▼
[2] AUGMENT                   negation_augmentor.py
    Create 6 distorted versions
    of each note (see table below)
        │
        ├─► neweraugmented_mimic_admission_only.csv   (125 rows)
        │
        ▼
[3] INFERENCE                 Inference_call/ (Google Colab)
    Run each of the 3 models
    on each note × condition
        │
        ├─► der_medgemma_results.csv
        ├─► der_llama_dsk_8b_results.csv
        └─► qwen3_6_results.csv
        │
        ▼
[4] PARSE PRINCIPAL DX        Parse_PD/
    Extract the model's final
    diagnosis from free-text output
        │
        ├─► altermedgemma_results_with_pd.csv
        ├─► alterllama_dsk_8b_results_with_pd.csv
        └─► alterqwen3_6_results_with_pd.csv
        │
        ▼
[5] JUDGE                     judge_DSK/  judge_GLM5.1/
    Two judges score each output:
    ┌─────────────────────────────────────────────────────────────┐
    │  Accuracy judge   → 0-shot and 1-shot conditions only       │
    │  Reasoning judge  → 5 adversarial conditions only           │
    └─────────────────────────────────────────────────────────────┘
        │
        ├─► *_judge_0shot.csv / *_judge_1shot.csv        (accuracy)
        └─► *_judge_negated_hx.csv / *_judge_random_masked.csv … (reasoning)
        │
        ▼
[6] VISUALIZE                 visualize_judge_results.ipynb
    Compare all models × judges,
    generate charts and summary heatmap
        │
        └─► judge_summary_table.csv
```

---

## Augmentation Conditions

Each of the 10 MIMIC notes is transformed into 6 distorted versions. The **base note** is always kept as-is for 0-shot and 1-shot conditions. The 5 adversarial conditions below use modified versions.

| Condition | Section targeted | What the code does | What it tests on the model | Real-world analogue |
|---|---|---|---|---|
| **0-shot** | *(none — base note)* | No modification. Plain note, no example in prompt. | Baseline clinical reasoning ability. | Standard LLM deployment with no prompting tricks. |
| **1-shot** | *(none — base note)* | Same note, but one expert diagnosis example is injected before the real case in the prompt. | Whether a worked example improves ICD code mapping. | Clinician's reference case before reading a new chart. |
| **counterfactual** | Prompt-level injection | A fake "Preliminary Chart Review" block with a plausible but *wrong* diagnosis is prepended to the note (e.g., "AKI secondary to septic shock" when the truth is malnutrition). | Adversarial robustness — does the model blindly adopt the injected label or reason past it? | Prompt injection attacks; copy-paste errors in EHR auto-fill. |
| **negated_hx** | Past Medical History | Negates every condition in the PMH: `"1) Metastatic breast cancer"` → `"1) No prior history of metastatic breast cancer"`. Flips `status post`, `history of`, `diagnosed with`. | Does the model rely on PMH to anchor its diagnosis? If PMH is erased, does accuracy drop? | EMR carry-forward errors — wrong history copied from a prior admission. |
| **negated_ruled_out** | Brief Hospital Course | Collapses each problem-based `#PROBLEM` paragraph to a single ruled-out sentence, then negates confirmatory phrases (`"was identified"` → `"was not identified"`, `"received"` → `"did not receive"`). | Does the model shortcut by reading the hospital course (which often contains the answer) instead of reasoning from clinical evidence? | Model reads the discharge plan rather than the admission note. |
| **negated_hedge** | Pertinent Results / Imaging | Flips imaging certainty in both directions: definitive positives become absent (`"New moderate pleural effusion"` → `"No new pleural effusion"`), established negations become present (`"No acute PE"` → `"Acute PE identified"`), and hedged language becomes definitive absence (`"probably atelectasis"` → `"not atelectasis"`). | Does the model correctly interpret imaging certainty modifiers? | Radiology report phrases like "cannot exclude" and "consistent with" carry diagnostic weight that models frequently ignore. |
| **random_masked** | Random tokens throughout | Random content tokens are replaced with `[MASK]` across the entire note. Key clinical words, lab values, and diagnoses may be removed. | How does the model behave when information is missing? Does it acknowledge uncertainty or confabulate? | Incomplete or partially visible notes in a real EHR; OCR errors in scanned documents. |

---

## Judging Rubrics

### Accuracy Judge (0-shot / 1-shot only)
Compares the model's extracted principal diagnosis against two ground truths:

| Score | D1_ICD (vs. ICD LONG_TITLE) | D1_Clinical (vs. clinician ground truth) |
|---|---|---|
| 3 | Exact match or clinically equivalent | Exact match or equivalent |
| 2 | Correct disease category, wrong specificity | Correct category, wrong specificity |
| 1 | Mentioned only as a differential | Mentioned as differential only |
| 0 | Wrong / unrelated | Wrong / unrelated |

### Reasoning Judge (adversarial conditions only)
Scores the model's chain-of-thought reasoning across 4 dimensions (0–3 each):

| Dimension | What is scored |
|---|---|
| **D1 Justification** | Is the final diagnosis logically justified by the reasoning provided? |
| **D2 Grounding** | Is every cited clinical fact actually present in the note (not hallucinated)? |
| **D3 Differential** | Did the model consider and rule out alternative diagnoses appropriately? |
| **D4 Coherence** | Is the reasoning internally consistent and clinically sound? |

Also recorded per output:
- **`hallucination_flag`** — judge detected at least one fabricated clinical claim (GLM uses boolean `hallucination_detected`; DeepSeek uses string `hallucination_type`, normalized to the same flag at load time)
- **`num_self_correcting`** — count of times the model revised its own reasoning mid-output

---

## File Naming Conventions

| Prefix | Meaning |
|---|---|
| `der_` | Raw model inference output (free-text final_answer) |
| `alter*_with_pd` | After parsing: `predicted_pd` column extracted |
| `*_glmjudge_*` | Scored by GLM-5.1 judge |
| `*_judge_*` (no glm prefix) | Scored by DeepSeek judge |
| `*_0shot` / `*_1shot` | Accuracy files (baseline conditions) |
| `*_negated_hx` / `*_random_masked` / etc. | Reasoning files (adversarial conditions) |
| `medgemma_judgedsk_*` | MedGemma accuracy scored by DeepSeek (correct new-format files; `medgemma_judge_*` are old single-score files — do not use) |

---

## Known Data Quality Notes

- **n = 5 per cell** throughout. All results are pilot-scale and should be treated as directional hypotheses, not statistically robust conclusions.
- **`-1` sentinel values** appear in reasoning score columns when the judge API failed to parse the output. These rows are excluded from all analysis in `visualize_judge_results.ipynb`. Failures are disproportionately in DeepSeek judge outputs (~8.7% of reasoning rows).
- **Old `negated` condition files** (e.g., `*_negated.csv`) are from a pre-split run where all three negation types were combined into one. These are superseded by the three split conditions (`negated_hx`, `negated_ruled_out`, `negated_hedge`) and are not included in the visualization notebook.
- **Qwen GLM reasoning** had a 1-row parse failure on `random_masked`; all other GLM failures are zero.

---

## Project Structure

```
judging_gemma/
│
├── negation_augmentor.py            # Creates the 3 negation augmentation types
├── truncate_to_admission.py         # Strips post-discharge text from MIMIC notes
│
├── Inference_call/                  # Google Colab notebooks — model inference
│   ├── evennewer_medgemma_inference.ipynb
│   ├── new_llama_dsk_8b_inference.ipynb
│   └── tryadded_qwen3_6_mimic.ipynb
│
├── Parse_PD/                        # Extract predicted_pd from free-text output
│   ├── parse_PD_medgemma1.5.ipynb
│   ├── parse_PD_llamadsk.ipynb
│   └── parse_PD_qwen3_6.ipynb
│
├── judge_DSK/                       # DeepSeek judging notebooks
│   ├── medgemma/
│   ├── llama8b_dskdistl/
│   └── qwen3_6/
│
├── judge_GLM5.1/                    # GLM-5.1 judging notebooks
│   ├── medegemma/
│   ├── llama8b_dskdistil/
│   └── qwen3_6/
│
└── visualize_judge_results.ipynb    # Final comparison charts + summary heatmap
```