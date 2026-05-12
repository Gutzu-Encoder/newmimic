# Can LLMs Reason From Clinical Evidence? A Pilot Evaluation of Principal Diagnosis Identification Under Adversarial Note Conditions

**Authors:** [your name]
**Date:** May 2026
**Dataset:** MIMIC-III (de-identified), n=5 per evaluation cell
**All figures:** see `visualize_judge_results.ipynb` — Chart numbers referenced throughout

---

## Abstract

We evaluate three large language models — MedGemma-1.5-4B, Llama-8B-DSK-Distill, and Qwen-3.6B — on their ability to identify the principal diagnosis from real de-identified hospital discharge notes. Rather than testing on clean notes alone, we systematically distort each note in five ways targeting specific clinical reasoning skills: reliance on medical history, shortcutting through the hospital course, interpretation of imaging certainty, handling of missing information, and resistance to adversarial injected diagnoses. Two independent LLM judges — DeepSeek-R1 and GLM-5.1 — score each output on both ICD coding accuracy and clinical reasoning quality across four dimensions. We find that Qwen-3.6B substantially outperforms the other two models on reasoning quality and hallucination suppression, but this advantage does not translate to meaningfully better ICD coding accuracy. All models show a consistent gap between clinical language understanding and formal ICD code mapping. Random masking of note content produces the highest hallucination rates, indicating that models confabulate rather than express uncertainty when clinical information is absent. These findings have direct implications for the safe deployment of LLMs in clinical documentation and coding workflows.

---

## 1. Introduction

Principal diagnosis identification — determining which condition was chiefly responsible for a patient's hospital admission — is a high-stakes clinical task that drives billing, resource allocation, and outcome reporting. ICD coding from free-text discharge notes is time-consuming and error-prone when done manually, making it a natural candidate for LLM-assisted automation. However, before such systems can be deployed safely, it is essential to understand not just whether models get the answer right, but *why* they fail when they do.

Existing benchmarks typically test models on clean, well-structured clinical notes and report a single accuracy number. This design cannot distinguish between a model that reasons carefully from clinical evidence and one that shortcuts by reading the discharge summary's conclusion section. It also cannot reveal how models behave when clinical notes are incomplete, when imaging findings are ambiguous, or when an authoritative-sounding but wrong diagnosis is injected into the prompt — all conditions that occur regularly in real clinical deployments.

This work addresses that gap. We construct a structured evaluation suite using real MIMIC-III discharge notes augmented under five adversarial conditions, each targeting a different component of clinical reasoning. We evaluate three models that represent distinct points on the medical LLM landscape: a purpose-built small medical model (MedGemma), a general-purpose distilled reasoning model (Llama-8B-DSK), and a mid-sized general instruction model (Qwen-3.6B). We score outputs with two independent LLM judges and analyze both accuracy and the quality of the reasoning chain that produced the answer.

---

## 2. Methods

### 2.1 Dataset

We use ten HADM_IDs from MIMIC-III, sampled across two clinical topics: oncology (neoplasm) and respiratory failure. Each patient's discharge note is truncated to the admission portion — everything after the final discharge plan is removed — to prevent models from reading the discharge diagnosis directly. Ground truth principal diagnoses are sourced from the MIMIC-III DIAGNOSES_ICD table and supplemented with clinician-written clinical descriptions to enable both ICD-code-level and clinical-language-level accuracy scoring.

### 2.2 Augmentation Conditions

Each note is presented to models under seven conditions. Two baseline conditions — zero-shot and one-shot — use the unmodified note. The one-shot condition prepends a single worked example of clinical reasoning from an unrelated case. Five adversarial conditions apply targeted distortions to specific sections of the note.

The negated_hx condition modifies the Past Medical History section, prepending "No prior history of" to every listed condition and flipping affirmative formulations such as "status post" and "history of." This probes whether models use PMH to anchor their diagnosis. The negated_ruled_out condition modifies the Brief Hospital Course, collapsing each problem-based paragraph into a ruled-out statement and negating confirmatory phrases throughout. This tests whether models shortcut by reading the clinical conclusion rather than reasoning from evidence. The negated_hedge condition modifies the Pertinent Results section, flipping imaging certainty in both directions: definitive findings become absent, established negations become present, and hedged language such as "probably" and "consistent with" becomes definitive absence. This probes whether models correctly weight radiology uncertainty modifiers. The random_masked condition replaces a random selection of content tokens throughout the note with the placeholder token [MASK], simulating incomplete or corrupted documentation. The counterfactual condition injects a fake "Preliminary Chart Review" block into the prompt that asserts a plausible but incorrect principal diagnosis, testing adversarial robustness.

### 2.3 Models

MedGemma-1.5-4B is Google's purpose-built medical vision-language model, fine-tuned on medical corpora. Llama-8B-DSK-Distill is a general-purpose model distilled from DeepSeek-R1 reasoning traces. Qwen-3.6B is a mid-sized general instruction-following model from Alibaba. All models receive the same system prompt instructing step-by-step clinical reasoning before stating the principal diagnosis.

### 2.4 Judges

Two LLM judges score each model output independently. DeepSeek-R1 and GLM-5.1 are used in separate judge pipelines. For the two baseline conditions (zero-shot and one-shot), judges score on two accuracy dimensions: ICD accuracy, comparing the model's predicted diagnosis against the ICD LONG_TITLE on a 0–3 scale, and clinical accuracy, comparing against the clinician-written ground truth description. For the five adversarial conditions, judges score reasoning quality across four dimensions — D1 Justification, D2 Grounding, D3 Differential, and D4 Coherence — and also record whether the output contains hallucinated clinical claims and how many times the model self-corrects during its reasoning chain.

A key normalization step is applied at data load time: GLM-5.1 represents hallucination detection as a boolean flag, while DeepSeek represents it as a string category where "none" means no hallucination detected. Both are converted to a unified boolean column before any analysis.

### 2.5 Data Quality

All n values are 5 per evaluation cell, reflecting the pilot scale of this study. Thirteen of 150 reasoning rows (8.7%) were excluded because the judge returned a -1 sentinel value, indicating output parse failure. Twelve of those thirteen failures came from the DeepSeek judge, concentrated in Qwen and MedGemma outputs under the random_masked and counterfactual conditions. This non-random failure pattern is a methodological concern discussed in the limitations section. All reported means are computed over valid (non-excluded) rows only.

---

## 3. Results

### 3.1 Accuracy on Baseline Conditions

Chart 1a and Chart 1b in the visualization notebook show ICD and clinical accuracy for all three models, separated by judge and by shot condition.

All three models score substantially lower on ICD accuracy than on clinical accuracy, regardless of which judge is used and regardless of whether the zero-shot or one-shot condition is applied. No model achieves a mean ICD score above 1.40 out of 3.00 in any single condition. Llama-8B-DSK achieves the highest mean ICD accuracy when averaged across both shots and both judges, followed closely by Qwen-3.6B, with MedGemma scoring lowest. On clinical accuracy the ordering reverses partially: Qwen-3.6B scores highest averaged across all conditions, with Llama-8B-DSK second and MedGemma lowest. The gap between ICD and clinical accuracy (Chart 1c) is largest for Qwen-3.6B, smallest for Llama-8B-DSK, and intermediate for MedGemma.

The one-shot condition does not produce consistent improvement across models. For MedGemma scored by DeepSeek, adding the example raises ICD accuracy substantially. For Qwen-3.6B scored by DeepSeek, the one-shot condition lowers ICD accuracy relative to zero-shot. For Llama-8B-DSK, scores are stable across shot conditions under both judges.

The most notable discrepancy between judges occurs for MedGemma at zero-shot: DeepSeek assigns a mean ICD score of 0.40, while GLM-5.1 assigns 1.20 for the same outputs. This three-fold difference on n=5 cases suggests the two judges apply different thresholds for what constitutes partial credit when a diagnosis is approximately but not exactly correct.

### 3.2 Judge Agreement

Chart 2a shows scatter plots of per-patient ICD scores from DeepSeek versus GLM-5.1 for all three models. Chart 2b shows mean absolute disagreement per model.

Exact score agreement between the two judges reaches 80% for all three models. Mean absolute disagreement is 0.30 points for Qwen-3.6B and 0.40 points for both Llama-8B-DSK and MedGemma. While 80% agreement appears high, it means one patient in five received different scores from the two judges. At n=5 per condition, a single disagreement constitutes a 20-percentage-point swing. The Qwen-3.6B result appears more reliable by this metric, but the clinical interpretation of disagreements — whether they reflect genuine judge bias or legitimate scoring ambiguity — cannot be resolved from the data alone.

### 3.3 Reasoning Quality Under Adversarial Conditions

Chart 3a shows mean scores on all four reasoning dimensions, broken down by model and judge. Chart 3b shows the D2 Grounding dimension specifically, broken down by adversarial condition and judge.

Qwen-3.6B achieves near-ceiling scores on all four dimensions under both judges. D1 Justification and D4 Coherence are both at the maximum of 3.00 under both judges. D2 Grounding reaches 3.00 under GLM-5.1 and 2.76 under DeepSeek. D3 Differential Diagnosis reaches 2.79 under GLM-5.1 and 2.41 under DeepSeek. These scores represent an exceptionally consistent performance across adversarial conditions.

Llama-8B-DSK and MedGemma score substantially lower on all dimensions. D3 Differential Diagnosis is the weakest dimension for both models under both judges, suggesting that considering and ruling out alternative diagnoses is the specific reasoning step these models handle least reliably. D2 Grounding is the second weakest, meaning models frequently cite clinical facts not supported by the note text. D1 Justification and D4 Coherence are comparatively stronger for both models, indicating that the models can construct a locally coherent narrative even when that narrative is not well-grounded in the note.

Breaking down D2 Grounding by adversarial condition, the counterfactual and random_masked conditions consistently produce the lowest grounding scores across all models. The negated_ruled_out condition produces the highest grounding scores, suggesting that even when the hospital course is distorted, models can still anchor to other sections of the note.

### 3.4 Hallucination Rates

Chart 4a shows hallucination rates by condition and judge. Chart 4b shows overall hallucination rates per model.

The contrast between Qwen-3.6B and the other two models is stark. Under GLM-5.1, Qwen-3.6B hallucinates in 4% of reasoning outputs. Under DeepSeek, the rate rises to 24%. Llama-8B-DSK hallucinates in 76% of outputs under GLM-5.1 and 61% under DeepSeek. MedGemma hallucinates in 60% of outputs under GLM-5.1 and 81% under DeepSeek — the highest rate observed in the dataset.

Breaking down by condition, the random_masked condition produces the highest overall hallucination rate at 79% averaged across all models and judges. The counterfactual condition produces the second highest at 64%. The structured negation conditions produce lower hallucination rates, with negated_ruled_out the lowest at 31%.

The 20-percentage-point gap between GLM-5.1 and DeepSeek on Qwen's hallucination rate (4% versus 24%) is the largest judge sensitivity difference in the dataset and warrants caution when interpreting either judge in isolation for this model.

### 3.5 Self-Correction Frequency

Chart 5a shows self-correction frequency by condition and judge. Chart 5b shows overall averages per model.

Qwen-3.6B self-corrects substantially more often than the other two models. Under GLM-5.1 Qwen averages 1.25 self-corrections per output, and under DeepSeek it averages 0.82. Llama-8B-DSK averages 0.48 under GLM-5.1 but drops dramatically to 0.09 under DeepSeek, a discrepancy whose cause is not clear from the data. MedGemma averages 0.24 under both judges, suggesting the model's self-correction rate is relatively stable across judge contexts.

---

## 4. Correlations Across Metrics

The first and strongest cross-metric relationship is between D2 Grounding and hallucination rate. Qwen-3.6B, which grounds its reasoning at near-ceiling levels, hallucinates in a small minority of cases. Llama-8B-DSK and MedGemma, which score in the 1.52–1.96 range on D2, hallucinate in the majority of cases. The relationship is monotonic: as grounding drops, hallucination rises. This is not a coincidence of model identity — it holds within the adversarial conditions as well, where the conditions that most depress D2 (counterfactual and random_masked) also produce the highest hallucination rates.

The second relationship is between self-correction frequency and reasoning quality. Qwen-3.6B's high self-correction rate aligns with its near-perfect reasoning scores. A model that revises its intermediate reasoning steps appears to arrive at better-grounded conclusions. Llama-8B-DSK's very low DeepSeek self-correction score (0.09) aligns with its lower coherence scores under that judge. Whether self-correction causes better reasoning or is merely a signature of a more capable model cannot be determined from this data, but the correlation is consistent across both judges.

The third relationship is between adversarial condition hardness and the structural nature of the information removed. Conditions that remove information entirely (random_masked) or replace it with wrong information (counterfactual) produce worse D2 grounding and higher hallucination than conditions that distort information in a structured way (negated_hx, negated_hedge, negated_ruled_out). This suggests that models are sensitive to the *presence* of clinical anchors in the text, not just their semantic content. A note with distorted but present clinical structure gives the model something to reason about; a note with random gaps invites confabulation.

The fourth relationship is between one-shot benefit size and baseline ICD accuracy. MedGemma, which shows the lowest baseline ICD accuracy under DeepSeek at zero-shot, shows the largest improvement from the one-shot example. Models that are already near a performance ceiling for this task format gain nothing from the example or regress slightly. This pattern is consistent with standard in-context learning behavior: demonstrations calibrate output format and granularity most when the model has no prior calibration for the task.

The fifth relationship is between the ICD-to-clinical accuracy gap and model size or capability tier. The gap grows from Llama-8B-DSK at 0.70 points to MedGemma at 0.75 points to Qwen-3.6B at 1.00 points. The more capable a model appears to be at clinical language understanding (as reflected in its clinical accuracy), the larger the residual gap to ICD code-level accuracy. This is counter-intuitive if one assumes that clinical understanding and ICD mapping improve together; it suggests they are partially dissociable skills.

---

## 5. Discussion

### 5.1 ICD Coding and Clinical Understanding Are Dissociable Skills

The consistent gap between ICD and clinical accuracy across all three models, combined with the finding that the gap grows with apparent clinical language capability, supports a view in which formal ICD coding requires a distinct skill from clinical comprehension. LLMs acquire clinical language understanding from the general distribution of medical text they encounter during pretraining. ICD code mapping, by contrast, requires memorization of an administrative taxonomy that is arbitrary, inconsistently applied in practice, and relatively rare in natural medical language. This dissociation has direct implications for system design: retrieving ICD codes from a structured coding database conditioned on a model's clinical understanding, rather than having the model generate codes directly, may be a more reliable architecture than end-to-end generation.

### 5.2 Qwen-3.6B's Reasoning Superiority Does Not Translate to ICD Accuracy

Qwen-3.6B dominates on every reasoning quality metric — D1 through D4, hallucination rate, and self-correction — but sits in the middle on ICD accuracy. The model reasons correctly about clinical presentation and produces well-grounded, internally consistent reasoning chains, but then fails at the final step of mapping to the correct ICD code. This localization of failure is informative: the bottleneck is not the reasoning process but the output representation. Investigating the specific cases where Qwen's reasoning is correct but the ICD mapping fails would clarify whether this is a prompting issue (the model would produce the right code if asked differently), a training coverage issue (low-frequency ICD codes are underrepresented in training), or a fundamental limitation of the mapping step.

### 5.3 Random Masking Is the Most Dangerous Adversarial Condition

Models hallucinate at the highest rate when information is randomly removed from the note rather than when it is distorted in structured ways. The clinical implication is direct: real-world hospital notes are frequently incomplete, contain redacted or missing sections, or arrive from external facilities with documentation gaps. A model that confabulates when data is absent poses a greater patient safety risk than one that simply gets the diagnosis wrong under adversarial negation. This argues for explicit uncertainty modeling in clinical LLM deployments — either through decoding-time abstention mechanisms or through judge-assisted flagging of low-grounding outputs before they are shown to clinicians.

### 5.4 LLM-as-Judge Reliability Is Insufficient at Pilot Scale

The 80% exact agreement between judges sounds reassuring until one considers that it is equivalent to one disagreement per five patients. The three-fold inter-judge discrepancy on MedGemma's zero-shot ICD score and the five-fold discrepancy on Qwen's hallucination rate (4% versus 24%) demonstrate that judge identity can dominate observed results at small n. Any future work using LLM-as-Judge as the primary evaluation signal in a clinical domain should establish inter-judge agreement on a held-out human-annotated sample before reporting judge scores as ground truth. This is particularly relevant for ACL and EMNLP submissions that use judge-based evaluation pipelines.

### 5.5 DeepSeek Judge Parse Failures Are a Non-Random Signal

Twelve of the thirteen judge failures (rows returning -1 on all scoring dimensions) came from the DeepSeek judge, and they were concentrated in Qwen and MedGemma outputs under the most adversarially challenging conditions. This is unlikely to be random. Long or structurally unusual model outputs — which occur more frequently when a model encounters a random_masked or counterfactual note — may exceed the judge's reliable parsing range. Excluding these rows, as this study does, is the methodologically conservative choice, but it introduces survivorship bias: the means reported for these conditions are computed over the outputs the judge could handle, not over all outputs. Future work should either use more robust judge prompting to reduce parse failures or separately report failure rates as a primary metric alongside quality scores.

---

## 6. Limitations

This study should be treated as a pilot with directional findings rather than a definitive benchmark. With n=5 per evaluation cell, every reported percentage corresponds to at most one patient flipping between categories. Effect sizes that appear large may not survive replication on a larger sample.

The adversarial augmentations are rule-based and approximate. The negation rules do not perfectly model the diversity of clinical note structures encountered in practice, and some augmentations may produce unnatural text that models can detect as distorted even without explicit instruction. A human review of augmented notes for naturalness is needed before these conditions can be used in a peer-reviewed benchmark.

The counterfactual condition uses a single injection template per patient. A more rigorous test of adversarial robustness would require multiple injection phrasings and varying degrees of plausibility to measure the injection's effect as a function of its persuasiveness.

Finally, all three models are evaluated in a single-pass zero-context setting. Real clinical deployment may involve retrieval augmentation, multi-turn dialogue with clinicians, or ensemble methods. The findings here apply specifically to single-pass instruction-following inference.

---

## 7. Conclusion

We find that Qwen-3.6B produces substantially better clinical reasoning chains and far fewer hallucinations than MedGemma or Llama-8B-DSK under all five adversarial note conditions, yet this reasoning advantage does not produce meaningfully better ICD coding accuracy. The consistent gap between clinical language understanding and formal ICD mapping across all models suggests these are partially independent skills that may require independent interventions to improve. The random_masked adversarial condition produces the highest hallucination rate of any condition tested, identifying model behavior under information absence as the primary safety concern for clinical deployment. Inter-judge disagreement at small n is large enough to affect qualitative conclusions, reinforcing the need for human annotation validation in LLM-judged clinical benchmarks.

---

*Pilot study — n=5 per evaluation cell. All quantitative claims are directional. Full visualization: `visualize_judge_results.ipynb`. Summary statistics: `judge_summary_table.csv`.*
