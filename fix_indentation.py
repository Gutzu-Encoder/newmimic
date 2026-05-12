"""Fix indentation in notebook cells after regex replacements."""
import json
from pathlib import Path

DL = Path(r"C:\Users\USER\Downloads\judging_gemma")

for fname in ['evennewer_medgemma_inference.ipynb', 'tryadded_qwen3_6_mimic.ipynb']:
    path = DL / fname
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        lines = cell['source']
        # Find first non-empty line to determine base indent
        base_indent = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                base_indent = len(line) - len(line.lstrip())
                break

        fixed = []
        for line in lines:
            stripped = line.lstrip()
            # If line has more indent than base but no content reason, fix it
            if stripped and len(line) - len(line.lstrip()) > base_indent + 4:
                # Reduce by 4 spaces
                fixed.append(line[4:])
            else:
                fixed.append(line)

        cell['source'] = fixed

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Fixed indentation in {fname}")

print("Done.")
