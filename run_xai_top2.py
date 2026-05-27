"""
Top-2 모델(efficientnetv2, nextvit)에 대해 GradCAM, Saliency, LIME, SHAP 순서로 실행
"""
import subprocess
import sys
import os

XAI_SCRIPTS = [
    "xai/generate_gradcam_only.py",
    "xai/generate_saliency_only.py",
    "xai/generate_lime_only.py",
    "xai/generate_shap_only.py",
]

python = sys.executable

for script in XAI_SCRIPTS:
    name = os.path.basename(script)
    print(f"\n{'#'*60}")
    print(f"  Running: {name}")
    print(f"{'#'*60}")
    result = subprocess.run([python, script], cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"[ERROR] {name} failed with return code {result.returncode}")
    else:
        print(f"[OK] {name} finished successfully")

print("\n" + "="*60)
print("All XAI scripts complete!")
print("="*60)
