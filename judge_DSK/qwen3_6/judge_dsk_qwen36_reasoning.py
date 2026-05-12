import sys
sys.path.insert(0, r"C:\Users\USER\Downloads\judging_gemma")
from judging_gemma.config import *
from judging_gemma.judging_utils import load_and_fix_gt, load_clinical_gt, parse_json_response, attach_cleaned_text

import json

import re

import time

import os

import pandas as pd

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

RESULTS_FILE = r"C:\Users\USER\Downloads\alterqwen3_6_results_with_pd.csv"
MIMIC_FILE   = r"C:\Users\USER\Downloads\neweraugmented_mimic_admission_only.csv"
MIMIC_FIXED  = r"C:\Users\USER\Downloads\augmented_data_mimic_fixed.csv"

OUTPUT_PATHS = {
    'negated_hx':        r"C:\Users\USER\Downloads\qwen36_judge_negated_hx.csv",
    'negated_ruled_out': r"C:\Users\USER\Downloads\qwen36_judge_negated_ruled_out.csv",
    'negated_hedge':     r"C:\Users\USER\Downloads\qwen36_judge_negated_hedge.csv",
    'random_masked':     r"C:\Users\USER\Downloads\qwen36_judge_random_masked.csv",
    'counterfactual':    r"C:\Users\USER\Downloads\qwen36_judge_counterfactual.csv",
}

MODEL = DSK_MODEL
client = OpenAI(
    api_key=API_KEY,
    base_url = DSK_BASE_URL
)

SYSTEM_PROMPT = """You are a clinical reasoning auditor.
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

Hallucination severity: 0=none, 1=minor/unlikely to affect diagnosis, 2=moderate/changes clinical picture, 3=severe/directly inverts key finding"""


def judge_reasoning(row):
    # real_note is only set for counterfactual; for all others use cleaned_text
    # row.get() finds the key even if value is NaN, so must check pd.notna()
    real_note = row.get('real_note')
    note_text = real_note if pd.notna(real_note) else row['cleaned_text']
    user_input = (
        f"### CLINICAL NOTE:\n{note_text}"
        f"\n\n### MODEL REASONING:\n{row['final_answer']}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_input}
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("Empty response from API")
        clean_content = re.sub(r"```json|```", "", content).strip()
        return json.loads(clean_content)
    except Exception as e:
        return {
            "d1_justification": -1, "d2_grounding": -1,
            "d3_differential": -1,  "d4_coherence": -1,
            "hallucination_type": "error", "hallucination_severity": -1,
            "audit_comment": str(e),
            "num_self_correcting": -1, "self_correcting": ""
        }
        
df_results = pd.read_csv(RESULTS_FILE)
df_mimic   = pd.read_csv(MIMIC_FILE)

mimic_fixed = pd.read_csv(MIMIC_FIXED)[['HADM_ID', 'ground_truth', 'SHORT_TITLE', 'LONG_TITLE']].drop_duplicates('HADM_ID')
df_results = df_results.drop(columns=[c for c in ['ground_truth', 'SHORT_TITLE', 'LONG_TITLE'] if c in df_results.columns])
df_results = df_results.merge(mimic_fixed, on='HADM_ID', how='left')

corrupted = ['negated_hx', 'negated_ruled_out', 'negated_hedge', 'random_masked', 'counterfactual']
df_cor = (
    df_results[df_results['condition'].isin(corrupted)]
    .drop_duplicates(subset=['HADM_ID', 'condition'], keep='first')
    .copy()
)

merged = attach_cleaned_text(df_cor, df_mimic)

for cond in corrupted:
    n = (merged['condition'] == cond).sum()
    has_text = merged[merged['condition'] == cond]['cleaned_text'].notna().sum()
    print(f"  {cond}: {n} rows, {has_text} with cleaned_text")
print(f"Total: {len(merged)} rows to audit")

final_records = []
for i, (_, row) in enumerate(merged.iterrows()):
    condition = row['condition']
    print(f"[{i+1}/{len(merged)}] [{condition}] HADM {row['HADM_ID']}...")
    audit = judge_reasoning(row)
    print(audit)
    combined = {**row.to_dict(), **audit}
    final_records.append(combined)
    time.sleep(0.5)

results_df = pd.DataFrame(final_records)

print("\n--- Results ---")
for cond in corrupted:
    df_c = results_df[results_df['condition'] == cond]
    if len(df_c) == 0:
        continue
    out_path = OUTPUT_PATHS[cond]
    df_c.to_csv(out_path, index=False)
    cols = ['d1_justification', 'd2_grounding', 'd3_differential', 'd4_coherence']
    avgs = df_c[cols].mean().round(2).to_dict()
    print(f"{cond} ({len(df_c)} rows) -> {out_path}")
    print(f"  {avgs}")
