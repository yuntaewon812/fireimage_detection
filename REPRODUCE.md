# 산불 감지 프로젝트 — 재현 가이드 (REPRODUCE)

> 이 문서 하나로 전체 실험을 동일하게 재현합니다.
> 작업이 로컬·Colab을 오갔으므로, **각 단계가 어디서 도는지**와 **정확한 명령**을 모두 기록합니다.

---

## 프로젝트 진행 흐름 (실험 히스토리)

> 각 실험을 **왜** 했는지의 맥락. 아래 "실행 순서"(§3)와 함께 보면 됩니다.

### 실험 A — 데이터·학습 기반 확립
- 데이터 누수(train/test 영상 겹침), 모델 전처리 오류 등 여러 문제를 해결 → **정제 데이터 완성**
- 정제 데이터로 baseline 학습 완료
- → 설정 A (baseline)

### 실험 B·C·D — 모델 성능 개선 시도 (OOD 개선 목표)
- 백본 교체, 데이터 조작(증강), Mixup 등으로 OOD(처음 보는 영상) 수치 개선 시도
- → 설정 B(pretrained) / C(+증강) / D(+Mixup)
- **그래도 OOD 수치 개선 안 됨**
- 원인 진단: **근본적인 데이터 불균형 + 분포 문제** (모델 기법만으론 한계)

### 실험 E — 데이터셋 재설계로 돌파
- 데이터셋을 다시 설계 → **분포를 고르게** 재구성
- OOD 수치 점검 → **OOD 개선 확인** (핵심 개선은 데이터 재설계에서 나옴)
- 후속 분석 진행:
  - **경량화** (INT8 양자화)
  - **XAI 4종** (GradCAM, SHAP, LIME, Saliency) — *fold0 가중치 기준* (설명 품질 비교가 목적)
  - **DropInc / PosNeg AUC** 진행 → 완료 → 결과 분석
  - **Sensitivity / Stability** 진행 → 완료 → 결과 분석
  - **Ablation(LOO) 설계 → 결과 분석**:
    **OOD에는 백본(사전학습)이 가장 큰 영향, 강증강은 오히려 역효과**

### 파이프라인 — 최적 모델로 재학습
- Ablation 결론 반영: **강증강을 뺀 구성(=설정 D / no_strongaug)으로 재학습**
- 이 가중치를 추론 파이프라인·데모 백본으로 사용 (maxvit + GradCAM 코어)

> **핵심 서사**: "모델 기법(B·C·D)으로는 OOD가 안 풀렸고, **데이터 재설계(E)**가 돌파구였다.
> 이후 Ablation으로 **사전학습이 핵심·강증강은 역효과**임을 입증, 강증강을 뺀 모델로 배포했다."

---

## 0. 재현에 고정된 핵심 설정 (절대 변경 금지)

| 항목 | 값 | 비고 |
|------|------|------|
| 랜덤 시드 | **1004** | 분할·학습 전체 동일 (교수님 지시) |
| 교차검증 | `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1004)` | 그룹=영상출처 |
| 학습 이미지 크기 | **160×160** | `models/*.py` 전부 img_size=160 |
| 배치 크기 | 8 | |
| 최대 epoch / patience | 15 / 5 | early stopping |
| 상위 2개 모델 | **maxvit, efficientnetv2** | 성능평가 결과 |

---

## 1. 환경 (로컬 & Colab 공통)

```bash
# Python 3.10 기준
pip install -r requirements.txt
# Colab 추가 (XAI/경량화 실행 시)
pip install timm grad-cam lime scikit-image scipy
```

- **로컬**: Windows 11, PyTorch CPU (학습은 비현실적 → 평가·XAI·데모만)
- **Colab**: L4 GPU (학습 전부 여기서)

---

## 2. 데이터

- **재현 기준 산출물**: `fireimage_clean.zip` (≈33MB, normal 792 / abnormal 727)
  - ⚠️ `download_data.py`는 사내 LAN(192.168.x.x) 주소라 외부 재현 불가 → **zip을 기준으로 사용**
- **압축 해제**
  - 로컬: zip을 `data/fireimage/` 에 풀기 (구조: `data/fireimage/{normal,abnormal}/...`)
  - Colab: `python colab_setup.py` (zip 자동 탐색·해제)

데이터 폴더 구조:
```
data/fireimage/
├── normal/    (정상 792장)
└── abnormal/  (화재 727장, youtube/ 등 하위 그룹 포함)
```

---

## 3. 실행 순서 (어디서 / 무엇을)

| 단계 | 작업 | 실행 위치 | 명령 / 노트북 |
|:---:|------|:---:|------|
| ① | 데이터 준비 | 로컬/Colab | `python colab_setup.py` (또는 zip 수동 해제) |
| ② | **누적 ablation 학습** (A~E, 7모델) | **Colab L4** | `demo/colab_ablation.ipynb` → `python main_ablation.py --setting all` |
| ③ | **LOO ablation 학습** (2모델, 단일시드) | **Colab L4** | `demo/colab_ablation_loo.ipynb` → `python main_ablation_loo.py --seeds 1004` |
| ④ | **no_strongaug 재학습** (배포용) | **Colab L4** | `demo/colab_retrain_no_strongaug.ipynb` |
| ⑤ | 성능평가 집계 | 로컬 | `python aggregate_loo.py` |
| ⑥ | XAI 평가 (PosNeg/DropInc) | 로컬 | `python aucGradSaliency.py` 등 6종 |
| ⑦ | XAI Sensitivity/Stability | 로컬 또는 Colab | `python sensitivity_stability.py` / `demo/colab_sens_stab.ipynb` |
| ⑧ | INT8 경량화 | 로컬/Colab | `python quantize_models.py` |
| ⑨ | 추론 파이프라인 | 로컬 | `inference/fire_pipeline.py` (모듈) |
| ⑩ | 데모 사이트 | 로컬 | `python demo/app.py` |

---

## 4. 단계별 상세

### ② 누적 Ablation 학습 (Colab L4)
설정 A(baseline)/B(pretrained)/C(+증강)/D(+mixup)/E(+강증강).
```bash
python main_ablation.py --class_name fireimage --setting all --epochs 15 --patience 5
```
- 저장: `model_save/fireimage_abl_{B,C,D,E}/`, 결과 `results/fireimage_abl_{...}/metrics.csv`
- 가중치 백업: `weights_E.zip` (설정 E)

### ③ LOO Ablation 학습 (Colab L4) — 핵심 실험
설정 E(Full)에서 요소 하나씩 제거.
```bash
python main_ablation_loo.py --variants full,no_pretrained,no_augment,no_mixup,no_strongaug \
                            --models efficientnetv2,maxvit --seeds 1004 --epochs 15 --patience 5
```
- variant별 저장: `model_save/fireimage_loo_{variant}_s1004/`
- 지표 + GradCAM Sens/Stab: `results/fireimage_loo_{variant}_s1004/{metrics,xai_sens_stab}.csv`

### ④ no_strongaug 재학습 (Colab L4) — 배포 모델
LOO 결과 "강증강 제거(=설정 D)"가 OOD 최고 → 배포용으로 이 config만 재학습.
```bash
python main_ablation_loo.py --variants no_strongaug --seeds 1004 --epochs 15 --patience 5
```
- 다운로드: `weights_no_strongaug.zip`
- **데모 연결**: 압축 해제 후 `model_save/fireimage_loo_no_strongaug_s1004/fold0/{maxvit,efficientnetv2}.pt`
  를 `model_save/fireimage/fold0/` 로 복사 (데모가 읽는 경로)

### ⑤ LOO 집계 + 통계 (로컬)
```bash
python aggregate_loo.py     # results_LOO/loo_summary.csv + 콘솔 표
```
- ΔOOD, paired t-test/Wilcoxon(참고), Cohen's d, GradCAM Sens/Stab
- 단일 시드 → "3-fold 일관성"으로 해석

### ⑥ XAI 평가 (로컬, fold0 가중치 기준)
```bash
python aucGradSaliency.py          # PosNeg AUC (GradCAM+Saliency)
python aucGradSaliencyPerturb.py
python aucLimeShap.py              # PosNeg AUC (LIME+SHAP(IG))
python aucLimeShapPerturb.py
python dropIncGradSaliency.py      # Drop/Increase AUC
python dropIncLimeShap.py
```
- 대상: `model_save/fireimage_abl_E/fold0/` (efficientnetv2, maxvit)
- 결과: `results_POS_NEG/`, `results_DROP_INC/`
- ⚠️ SHAP는 `shap.GradientExplainer` 호환 문제로 **Integrated Gradients**로 대체 (논문 표기: "IG")

### ⑦ Sensitivity / Stability (로컬 또는 Colab)
```bash
python sensitivity_stability.py    # results_SENS_STAB/sens_stab_{cpu,cuda}.csv
```
- Sensitivity = E(x) vs E(x+noise) L2 / Stability = 반복 설명 std (둘 다 낮을수록 좋음)
- CPU vs L4 시간 비교는 `demo/colab_sens_stab.ipynb`

### ⑧ INT8 경량화 (로컬/Colab)
```bash
python quantize_models.py          # model_save/fireimage_abl_E_quant/, results/quantization_report.csv
```

### ⑩ 데모 사이트 (로컬)
```bash
python demo/app.py                 # Gradio, share 링크 생성
# 풍향 자동조회: set OPENWEATHER_API_KEY=<키>
```
- 상위 2모델(maxvit, efficientnetv2) **교차검증 앙상블**: 두 모델 일치 → 신뢰도 높음
- 가중치는 `model_save/fireimage/fold0/` 에서 로드 (④에서 배치)

---

## 5. Colab 노트북 목록 (재현용)

| 노트북 | 용도 |
|--------|------|
| `demo/colab_ablation.ipynb` | 누적 ablation 학습 (A~E, 7모델) |
| `demo/colab_ablation_loo.ipynb` | LOO ablation (2모델, 단일시드 1004) |
| `demo/colab_retrain_no_strongaug.ipynb` | 배포 모델(no_strongaug) 재학습 |
| `demo/colab_sens_stab.ipynb` | Sensitivity/Stability + CPU vs L4 시간 |

GitHub에서 바로 열기: `https://colab.research.google.com/github/yuntaewon812/fireimage_detection/blob/main/demo/<노트북>`

---

## 6. 주요 결과 요약 (재현 시 기대값)

### LOO Ablation (ΔOOD, full 대비)
| 제거 요소 | efficientnetv2 | maxvit |
|------|:---:|:---:|
| no_pretrained | −0.082 | −0.106 (🔴 필수) |
| no_strongaug | +0.018 | +0.049 (🟢 빼야 좋음, p=0.029) |

→ **결론: 사전학습은 필수, 강증강은 OOD에 역효과. 배포 모델 = maxvit + no_strongaug (OOD 0.954).**

### XAI (maxvit 기준, 파이프라인용)
- 코어 XAI = **GradCAM** (영역 연산에 적합 + 지연 예산 충족)
- 검증용 = SHAP(IG) (가장 일관적이나 느림 → 오프라인)

---

## 7. 재현 시 주의 (정직한 한계)

1. **데이터 원본**(download_data.py)은 LAN 전용 → `fireimage_clean.zip` 으로만 재현
2. **학습은 GPU 필수** — 로컬 CPU로는 비현실적, Colab L4 사용
3. **LOO는 단일 시드(1004)** — "통계적 유의" 대신 "3-fold 일관성"으로 보고
4. **XAI 평가는 fold0 가중치만 사용** — 이유:
   - XAI(GradCAM/SHAP/LIME/Saliency, DropInc/PosNeg, Sens/Stab)는 **"설명이 불 영역을 잘 짚는가"를 방법 간 비교**하는 게 목적
   - 일반화 성능(OOD) 측정이 아니므로 **모델 1개(fold0)면 방법 비교가 성립** → fold 전체 불필요
   - 실무적으로도 로컬에 fold0 가중치가 있어 즉시 평가 가능했음
   - (참고: OOD 성능 측정인 LOO ablation은 fold0/1/2 **3개 모두** 사용)
5. 가중치 백업본: `model_save_backup.zip`(원본 v25), `weights_E.zip`(설정 E)
