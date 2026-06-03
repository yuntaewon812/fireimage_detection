# 🔥 산불 감지·대응 데모 (Gradio)

학습된 모델 + `inference/fire_pipeline.py` 를 그대로 활용해, 이미지/영상을 입력하면
화재 분류 → GradCAM → 연기방향·풍향 분석 → **VLM 산불 대응 보고**를 생성하는 데모.

## 설계 핵심

- **상위 2개 모델 교차검증**: 성능평가 1·2위 모델로 동시 추론 → 일치 여부로 신뢰성 판정
  - 두 모델 모두 화재 → 신뢰도 높음
  - 불일치 → 주의 플래그(추가확인 권고)
- **영상 입력 → 정확한 연기방향**: 연속 프레임 Optical Flow로 실제 이동벡터 추정
  (이미지는 GradCAM 모양 기반 PCA 추정 — 덜 정확)
- **VLM 대응 보고**: 강도·연기색·방향을 근거로 산불 상태/위치/대응단계/장비·인력/주의 생성

## 파일 구조

```
demo/
  app.py            # Gradio UI + 2모델 교차검증 오케스트레이션
  model_loader.py   # .pt 가중치 로드 + 상위모델 자동 랭킹(metrics.csv)
  vlm_responder.py  # 연기색/강도/자원 분석 + VLM 대응텍스트 생성
  map_utils.py      # EXIF GPS 추출 + folium 발화점/확산방향 지도
  requirements.txt
  README.md
```

## 데이터 흐름

```
이미지/영상 ─┬─ 모델A ─ pipeline.run() ─ 결과A ─┐
             └─ 모델B ─ pipeline.run() ─ 결과B ─┤
                                                ├─ 교차검증(일치/신뢰도)
                                                ├─ VLM 대응 보고 생성
                                                └─ GradCAM·마스크·지도 시각화
```

## 실행

```bash
pip install -r demo/requirements.txt          # 본체 requirements.txt 위에
# (선택) 고품질 VLM 보고: Claude 비전
set ANTHROPIC_API_KEY=sk-...                   # Windows
python demo/app.py                             # share=True → 공유링크 자동
```

VLM 키가 없으면 근거 데이터 기반 텍스트로 자동 폴백(동적 값 반영).

## 학습 완료 후 할 일

1. `results/fireimage/metrics.csv` 생성 확인
2. `model_loader.rank_top_models(2)` 가 자동으로 F1 상위 2개 선정
   (수동 고정하려면 `model_loader.TOP_MODELS` 편집)
3. `model_save/fireimage/fold0/{모델}.pt` 가중치 배치

## 향후 확장

- 지형(경사)·연료(임상)·수원 접근성 → 지도 API/DEM 연동
- 실시간 웹캠/CCTV 스트림 (`gr.Image(sources=["webcam"])`)
- 확산 예측 시뮬레이션 (풍향+경사+연료)
```
