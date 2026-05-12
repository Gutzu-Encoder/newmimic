import pandas as pd, re

CUTOFF_PATTERNS = [
    r'Brief Hospital Course',
    r'HOSPITAL COURSE',
    r'Discharge Diagnoses',
    r'DISCHARGE DIAGNOSES',
    r'Discharge Condition',
]

def truncate_at_hospital_course(text: str) -> str:
    for pat in CUTOFF_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return text[:m.start()].strip()
    return text  # no cutoff found, return full

df = pd.read_csv(r"C:\Users\USER\Downloads\neweraugmented_mimic.csv")
df['admission_text'] = df['cleaned_text'].apply(truncate_at_hospital_course)
df.to_csv(r"C:\Users\USER\Downloads\neweraugmented_mimic_admission_only.csv", index=False)

print(df[['HADM_ID','augmentation_type']].value_counts())
print("\nSample truncation for HADM 176830:")
sample = df[df['HADM_ID']==176830].iloc[0]
print(f"Full: {len(sample['cleaned_text'])} chars → Truncated: {len(sample['admission_text'])} chars")