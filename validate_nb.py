import json
with open(r'C:\fireimage_detection\demo\colab_ablation.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
print(f'셀 수: {len(nb["cells"])}, nbformat: {nb["nbformat"]}')
print('JSON 유효 OK')
