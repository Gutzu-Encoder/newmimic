import re, json, time, pandas as pd
from .config import MIMIC_FIXED, DECEPTIVE_INJECTION, MIMIC_AUGMENTED, CLINICAL_GT, NEGATION_TYPES

def load_and_fix_gt(df: pd.DataFrame) -> pd.DataFrame:
    """Drop wrong ground-truth cols, merge from fixed file. Call on every notebook."""
    fixed = (
        pd.read_csv(MIMIC_FIXED)[['HADM_ID', 'ground_truth', 'SHORT_TITLE', 'LONG_TITLE']]
        .drop_duplicates('HADM_ID')
    )
    df = df.drop(columns=[c for c in ['ground_truth','SHORT_TITLE','LONG_TITLE'] if c in df.columns])
    return df.merge(fixed, on='HADM_ID', how='left')

def load_clinical_gt(df: pd.DataFrame) -> pd.DataFrame:
    """Merge clinical_gt and structural difficulty columns from clinical_ground_truth.csv."""
    gt = pd.read_csv(CLINICAL_GT)
    gt.columns = gt.columns.str.strip()
    gt = gt[['HADM_ID', 'clinical_gt', 'structural']].drop_duplicates('HADM_ID')
    df = df.drop(columns=[c for c in ['clinical_gt', 'structural'] if c in df.columns])
    return df.merge(gt, on='HADM_ID', how='left')

def parse_json_response(content: str) -> dict:
    return json.loads(re.sub(r"```json|```", "", content).strip())

def attach_cleaned_text(df_cor: pd.DataFrame, df_mimic: pd.DataFrame) -> pd.DataFrame:
    """Merge cleaned_text for negated/random_masked/new-negation types; reconstruct for counterfactual.

    Handles:
    - negated, random_masked (legacy)
    - negated_hx, negated_ruled_out, negated_hedge (C3 new types)
    - counterfactual (B3 fix: stores real_note separately, does NOT inject into cleaned_text)
    """
    neg_conditions = ['negated', 'random_masked'] + NEGATION_TYPES
    df_nm = df_cor[df_cor['condition'].isin(neg_conditions)]
    merged_nm = pd.merge(
        df_nm,
        df_mimic[['HADM_ID','augmentation_type','cleaned_text']],
        left_on=['HADM_ID','condition'], right_on=['HADM_ID','augmentation_type'],
        how='left'
    )
    df_cf = df_cor[df_cor['condition'] == 'counterfactual'].copy()
    base = df_mimic.groupby('HADM_ID')['cleaned_text'].first().reset_index()
    df_cf = df_cf.merge(base, on='HADM_ID', how='left')
    # Store real note separately; judge ONLY sees real note (fixes B3)
    df_cf['real_note'] = df_cf['cleaned_text']
    df_cf['injected_text'] = DECEPTIVE_INJECTION + "Clinical note:\n" + df_cf['cleaned_text']
    # cleaned_text = real note only (B3 fix applied here)
    return pd.concat([merged_nm, df_cf], ignore_index=True)
