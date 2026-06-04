# 데모 사이트 인수인계

> 산불 감지·대응 Gradio 데모. 집 컴퓨터에서 이어서 작업할 때 이 문서부터 읽을 것.

## 한 줄 요약

`inference/fire_pipeline.py`(기존 추론 엔진)를 Gradio UI로 감싼 데모.
**성능평가 상위 2개 모델로 교차검증** → GradCAM·연기방향·풍향 분석 → **VLM 산불 대응 보고** 생성.

## 파일 구조 (모두 `demo/` 안)

| 파일 | 역할 |
|------|------|
| `app.py` | Gradio UI + 2모델 교차검증 오케스트레이션 (진입점) |
| `model_loader.py` | `.pt` 가중치 로드 + metrics.csv 기반 상위2개 자동 랭킹 |
| `vlm_responder.py` | 연기색/강도/자원 분석 + VLM 대응 텍스트 생성 |
| `map_utils.py` | EXIF GPS 추출 + folium 발화점·확산방향 지도 |
| `smoke_test.py` | Gradio 없이 배관 검증 (이미지 1장) |
| `make_sample_videos.py` | 샘플 이미지로 테스트 영상 3종 생성 (data/ 없이도 동작) |
| `debug_webcam.py` | 웹캠 컴포넌트 단독 테스트용 |
| `test_assets/` | 테스트 영상·샘플 이미지 (git 포함) |

## 실행

```bash
pip install -r demo/requirements.txt
python demo/app.py          # http://127.0.0.1:7860
```

VLM 고품질 보고 원하면(선택): `set ANTHROPIC_API_KEY=sk-...` 후 실행.
키 없으면 근거 데이터 기반 텍스트로 자동 폴백(동작에 지장 없음).

풍향 자동조회 원하면(선택): `set OPENWEATHER_API_KEY=...` 후 실행.

## UI 구성

- **영상 탭**: mp4 업로드 → 분석 실행 버튼 클릭
- **웹캠 탭**: 웹캠 촬영 → 즉시 분석 / 자동 분석(2초) 체크박스
- 이미지 탭 제거됨 (영상/웹캠만 사용)

## 데이터 흐름

```
영상/웹캠 ─┬ 모델A → fire_pipeline.run() → 결과A ┐
           └ 모델B → fire_pipeline.run() → 결과B ┤
                                                 ├ 교차검증(일치=신뢰↑/불일치=주의)
                                                 ├ VLM 대응보고
                                                 └ GradCAM·마스크·지도
```

## 동작 검증 상태 (2026-06-04 기준)

- ✅ 배관 전체 동작 확인
- ✅ 영상 입력, GradCAM/마스크 시각화, VLM 보고, 교차검증 UI
- ✅ GradCAM 크기 불일치 버그 수정 (160×160 → 원본 크기 리사이즈)
- ✅ fire_mask 크기 불일치 버그 수정 (vlm_responder, app.py 양쪽)
- ✅ 웹캠 탭 추가 (촬영 즉시 분석 + 2초 자동분석 타이머)
- ✅ 테스트 영상 3종 생성 스크립트 (`make_sample_videos.py`)
- ⚠️ 웹캠: 하드웨어 카메라 없으면 `NotFoundError` — USB 웹캠 또는 스마트폰 앱(DroidCam) 필요

## ⚠️ 알아둘 핵심 사항

1. **이중 softmax 수정됨**: `DeepModel`(ResNet/DenseNet)은 forward 끝에 Softmax를 적용하는데
   파이프라인이 또 softmax를 걸어 신뢰도가 뭉개진다. `model_loader.load_model()`에서
   로드 시 `model.activation`을 `Identity`로 교체해 해결.

2. **연기방향**: 영상은 Optical Flow로 실제 이동벡터 추정.
   OF 이동량 부족 시 PCA로 폴백(설계 의도 — 제거하지 말 것).

3. **경로 라우팅**: `fire_prob<0.5 → normal 즉시종료`.
   `fire_prob≥0.5 → GradCAM → HSV 화염색≥0.25면 flame, 아니면 smoke`.

4. **VLM**: `ANTHROPIC_API_KEY` 없으면 폴백 텍스트(규칙 기반) 자동 사용.
   4개 섹션(🔴상태/📍위치/🚁대응/⚠️주의)은 키 없이도 동적으로 생성됨.

## 테스트 자산 (git 포함)

```
demo/test_assets/
├── smoke_test.mp4         ← 원본 화재 연기 영상
├── test_fire_smoke.mp4    ← 검은연기+흰연기 합성 (abnormal)
├── test_fire_flame.mp4    ← 화염 합성 (abnormal)
├── test_normal.mp4        ← 파란하늘+녹색들판 합성 (normal, 흰연기 혼동 없음)
└── samples/               ← 정지 이미지 샘플 5종
```

## 남은 작업 (TODO)

1. **학습 완료된 가중치 연결** ← 최우선
   - Kaggle 커널 `yuntarwon/fireimage-training` Output에서 `model_save_backup.zip` 다운로드
   - 압축 풀어 `model_save/fireimage/fold0/*.pt` 배치 (git 제외 대상 → 수동 전송)
   - 가중치 없으면 무작위 초기화 → 화재/정상 예측이 랜덤임

2. **상위 2개 모델 확정**
   - `results/fireimage/metrics.csv` 이미 있음
   - `model_loader.rank_top_models(2)`가 F1 상위2개 자동 선정 (가중치 있어야 동작)
   - 수동 고정하려면 `model_loader.TOP_MODELS` 편집

3. (선택) ALERTCalifornia 공개 산불 CCTV 스트림 연동 — 실시간 테스트용
4. (선택) 실시간 웹캠/CCTV: 웹캠 하드웨어(또는 DroidCam) 필요

## 주의: git에 없는 것

- `data/` (Kaggle 데이터셋 `yuntarwon/fireimage-*` 에 있음)
- `model_save/` 가중치 (Kaggle 학습 Output에서 받아 수동 배치)
