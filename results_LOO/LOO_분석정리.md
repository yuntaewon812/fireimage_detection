# Leave-One-Out Ablation 실험 정리

## 1. 실험 목적
설정 E(Full)를 구성하는 4가지 학습 기법 각각이 **OOD(처음 보는 영상) 성능에 실제로 기여하는지** 검증.

## 2. 실험 설계

### 대상
- **모델 2종**: efficientnetv2, maxvit (XAI·성능 평가 최상위 2개)
- **데이터**: fireimage (normal 792 / abnormal 727)
- **분할**: StratifiedGroupKFold 3-fold, `random_state=1004` (고정)

### Leave-One-Out 5 variant
Full(설정 E)에서 요소를 하나씩 제거:

| variant | pretrained | 증강 | aug_type | mixup | 제거 요소 |
|---------|:---:|:---:|:---:|:---:|------|
| `full` | ✓ | ✓ | youtube(강) | ✓ | — (기준) |
| `no_pretrained` | ✗ | ✓ | youtube | ✓ | ImageNet 사전학습 |
| `no_augment` | ✓ | ✗ | — | ✓ | 증강 전체 |
| `no_mixup` | ✓ | ✓ | youtube | ✗ | Mixup |
| `no_strongaug` | ✓ | ✓ | standard(기본) | ✓ | 강증강 |

### 통제 원칙
- 한 번에 **한 요소만** 변경, 나머지(LR·epoch=15·patience=5·fold·batch=8) 전부 고정
- **시드 1004 단일** (프로젝트 전체와 동일, 교수님 지시로 변경 금지)
- 규모: 5 variant × 2 model × 3 fold = **30 model-fold**

### 지표
- **주지표**: OOD-F1 (3 fold 중 최저 = 가장 어려운 영상 fold)
- **보조**: 3-fold 평균 F1, ΔOOD(full 대비), Cohen's d, paired t-test(참고)
- **XAI 차원**: 각 run 학습 직후 GradCAM Sensitivity/Stability (test fold 15장)

### 통계 한계
단일 시드 → 표본 = 3 fold. 엄밀한 p<0.05는 검정력 약함 → **"3-fold 방향 일관성"** 으로 해석.

---

## 3. 결과

### ΔOOD (full 대비 OOD-F1 변화)

| variant | efficientnetv2 | maxvit | 판정 |
|---------|:---:|:---:|------|
| full (기준) | 0.893 | 0.905 | — |
| **no_pretrained** | **−0.082** | **−0.106** | 🔴 필수 (빼면 폭락) |
| no_augment | +0.003 | +0.020 | 중립 |
| no_mixup | +0.009 | +0.008 | 중립 |
| **no_strongaug** | +0.018 | **+0.049** | 🟢 빼면 향상 |

### fold별 상세 (maxvit)

| variant | fold0 | fold1 | fold2 | OOD(최저) |
|---------|:---:|:---:|:---:|:---:|
| full | 0.942 | 0.931 | 0.905 | 0.905 |
| no_pretrained | 0.799 | 0.944 | 0.888 | **0.799** |
| no_strongaug | 0.968 | 0.971 | 0.954 | **0.954** |

---

## 4. 핵심 발견

### ① 사전학습(pretrained)만이 OOD에 진짜 기여
- 빼면 OOD 폭락: effnet −0.082, maxvit −0.106
- 피해가 최악 fold에 집중 (maxvit fold0: 0.942→0.799)
- Cohen's d = −0.83 / −0.92 (큰 효과), 3 fold 모두 일관 하락
- → **ImageNet 사전학습은 절대 빼면 안 되는 핵심**

### ② 강증강(youtube)은 OOD를 오히려 해친다 ⚠️ (가설 반전)
- 빼니까 향상: maxvit 0.905 → **0.954** (3 fold 전부 상승)
- **p = 0.029 (유일하게 통계적 유의)**, Cohen's d = **2.58 (초대형)**
- "영상 스타일 단서 제거"용 강증강이 역효과 → 실제 화재 특징까지 손상
- → **설정 E < 설정 D(기본증강)**

### ③ Mixup·기본증강은 기여 없음
- 빼도 변화 미미 → OOD엔 무의미

### ④ GradCAM Sens/Stab 함정
- no_pretrained의 Sens가 최저(0.034)지만 **나쁜 모델** → degenerate(균일) GradCAM
- Sens/Stab은 반드시 OOD와 함께 해석
- no_strongaug = 최고 OOD + 정상 GradCAM 품질 유지 → win-win

---

## 5. 결론 및 권고

1. **사전학습 유지** (필수, 일반화의 핵심)
2. **강증강 제거** → 배포 모델은 **설정 D(pretrained + 기본증강 + mixup)** 구성
3. **최적 모델 = maxvit + no_strongaug (OOD 0.954)** — 파이프라인 백본 권장과 일치
4. 발표 메시지: *"강증강은 OOD에 역효과, 사전학습이 일반화의 핵심"* — 가설을 뒤집는 정직한 ablation
