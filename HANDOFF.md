# 인수인계 문서 — 화재 이미지 탐지 모델 성능평가

## 프로젝트 개요

화재 이미지 이진 분류(normal=비화재 / abnormal=화재) 7개 모델 성능 비교 실험.
데이터 누수를 수정한 뒤 전체 재평가를 진행 중이며, 현재 컴퓨터에서 Fold 0 학습이
부분 완료된 상태로 이 컴퓨터에서 나머지를 이어서 학습해야 한다.

---

## 데이터셋

```
data/fireimage/
    normal/          ← 비화재 이미지 (youtube 서브폴더 포함)
    abnormal/        ← 화재 이미지 (black_smoke, flame, white_gray_smoke, youtube, chimney_smoke 등)
```

- normal: 약 6,106장 / abnormal: 약 2,871장 (합계 ~8,977장)
- YouTube 폴더에는 연속 프레임이 있음 (예: `9UONERkDWq0_f000000.jpg`, `_f000005.jpg`, ...)
- **데이터 누수 방지**: StratifiedGroupKFold 적용 — 같은 YouTube 영상 프레임은 반드시 같은 fold에만 배치

---

## 7개 모델

| 모델 키 | 클래스 |
|---------|--------|
| Resnet50 | DeepModel('ResNet50') |
| DenseNet121 | DeepModel('DenseNet121') |
| efficientnetv2 | EfficientNetV2ForImageClassification |
| efficientnetv2_proposal | EfficientNetV2ForImageClassification_v2 |
| nextvit | NextViTForImageClassification |
| maxvit | MaxViTForImageClassification |
| internimage | InternImageForImageClassification |

---

## 평가 방법

- Stratified 3-Fold CV (StratifiedGroupKFold, random_state=1004)
- 지표: Accuracy / Loss / Recall / Precision / F1 / AUROC / ECE
- ECE(Expected Calibration Error): 확률 보정 품질, 낮을수록 좋음

---

## 현재 학습 진행 상황

### 완료된 가중치 (model_save/fireimage/fold0/)

| 모델 | 저장 경로 | 상태 |
|------|-----------|------|
| Resnet50 | fold0/Resnet50.pt | ✅ 완료 |
| DenseNet121 | fold0/DenseNet121.pt | ✅ 완료 |
| efficientnetv2 | fold0/efficientnetv2.pt | ✅ 완료 |
| efficientnetv2_proposal | fold0/efficientnetv2_proposal.pt | ✅ 완료 |
| nextvit | fold0/nextvit.pt | ✅ 완료 |
| maxvit | fold0/maxvit.pt | ✅ 완료 |
| internimage | — | ❌ 미완료 |

### 남은 작업 (15 runs)

- fold0: internimage 1개
- fold1: 7개 전체
- fold2: 7개 전체

fold1/2의 Resnet50·DenseNet121 구버전 가중치는 이미 삭제됨 (StratifiedKFold로 학습된 무효 가중치였음).

---

## 코드 핵심 설정 (main_v2.py)

```python
FORCE_RETRAIN = set()   # 빈 셋 — fold0 완료 가중치 유지
epochs   = 30           # (기존 60에서 축소)
patience = 8            # (기존 12에서 축소)
```

**주의**: FORCE_RETRAIN을 건드리지 말 것.
fold0에 저장된 가중치가 있으면 자동 스킵, 없으면 자동 학습.

---

## 환경 설정 및 실행 방법

### 1. Python 환경 (Python 3.9 ~ 3.11 권장)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm transformers einops
pip install grad-cam lime shap
pip install numpy pandas scikit-learn scipy
pip install opencv-python Pillow matplotlib seaborn umap-learn tqdm
```

또는 한 번에:
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> GPU가 CUDA 12.x라면 cu118 대신 cu121 사용:
> `--index-url https://download.pytorch.org/whl/cu121`

### 2. 학습 실행

```bash
cd C:\fireimage_detection   # 또는 프로젝트 폴더 경로
python main_v2.py --class_name fireimage
```

로그는 자동으로 저장됨:
- `training_stdout.txt`
- `training_stderr.txt`

### 3. 학습 완료 후 비교표 생성

```bash
python compare_models.py
```

결과 저장 위치:
- `results/fireimage/metrics.csv` — 전체 성능 수치
- `results/fireimage/model_comparison.png` — 비교 차트
- `results/fireimage/model_radar.png` — 레이더 차트

---

## 주의사항

- **InternImage**: 이전 실험에서 F1=0.405로 수렴 실패 이력 있음. LR=5e-5로 낮게 설정됨.
  학습 후 F1이 0.4~0.5 수준이면 수렴 실패로 판단.
- **Precision is ill-defined 경고**: 초반 epoch에서 정상 발생, 무시해도 됨.
- stdout이 비어 있어도 정상 (Python 출력 버퍼링). model_save 폴더에 .pt 파일이 쌓이면 진행 중인 것.
- 외부 검증셋이 있다면 `data/external_val/fireimage/normal/`, `abnormal/`에 넣으면 학습 완료 후 자동 평가됨.

---

## 파일 구조 요약

```
fireimage_detection/
├── main_v2.py              ← 메인 실행 스크립트
├── compare_models.py       ← 학습 완료 후 비교표 생성
├── evaluate_external.py    ← 외부 검증셋 평가
├── requirements.txt
├── HANDOFF.md              ← 이 문서
├── models/
│   ├── deep_model.py       ← ResNet50, DenseNet121
│   ├── efficientnetv2.py
│   ├── nextvit.py
│   ├── maxvit.py
│   └── internimage.py
├── utils/
│   ├── utils.py            ← 데이터 로드, 그룹 ID 부여 (누수 방지 핵심)
│   └── train_loop.py       ← 학습 루프, ECE 계산
├── data/
│   └── fireimage/
│       ├── normal/
│       └── abnormal/
├── model_save/
│   └── fireimage/
│       ├── fold0/          ← 6개 완료된 가중치 있음
│       ├── fold1/          ← 비어있음 (학습 필요)
│       └── fold2/          ← 비어있음 (학습 필요)
└── results/
    └── fireimage/
        └── metrics.csv     ← 기존 결과 (이번 재학습으로 갱신 예정)
```
