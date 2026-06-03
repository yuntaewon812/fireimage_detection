# 데모 사이트 인수인계

> 산불 감지·대응 Gradio 데모. 학교 컴퓨터에서 이어서 작업할 때 이 문서부터 읽을 것.

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
| `make_test_video.py` | YouTube 연속프레임 → 테스트 .mp4 + Optical Flow 검증 |
| `test_assets/` | 테스트 영상·샘플 이미지 (git 포함) |

## 실행

```bash
pip install -r demo/requirements.txt
python demo/app.py          # http://127.0.0.1:7860
```

VLM 고품질 보고 원하면(선택): `set ANTHROPIC_API_KEY=sk-...` 후 실행.
키 없으면 근거 데이터 기반 텍스트로 자동 폴백(동적 값 반영, 동작에 지장 없음).

## 데이터 흐름

```
이미지/영상 ─┬ 모델A → fire_pipeline.run() → 결과A ┐
            └ 모델B → fire_pipeline.run() → 결과B ┤
                                                  ├ 교차검증(일치=신뢰↑/불일치=주의)
                                                  ├ VLM 대응보고
                                                  └ GradCAM·마스크·지도
```

## 동작 검증 상태 (2026-06-04 기준)

- ✅ 배관 전체 동작 확인 (`smoke_test.py`, Gradio HTTP 200)
- ✅ 이미지/영상 입력, GradCAM/마스크 시각화, VLM 보고, 교차검증 UI
- ✅ 영상 입력 시 Optical Flow 연기방향 (연기 가시 구간 필요)

## ⚠️ 알아둘 핵심 사항

1. **이중 softmax 수정됨**: `DeepModel`(ResNet/DenseNet)은 forward 끝에 Softmax를 적용하는데
   파이프라인이 또 softmax를 걸어 신뢰도가 뭉개진다. `model_loader.load_model()`에서
   로드 시 `model.activation`을 `Identity`로 교체해 해결(= eval과 동일한 softmax(raw) 사용).

2. **연기방향엔 영상 필요**: 단일 이미지는 PCA 모양추정(부정확).
   연속 프레임(영상)이라야 Optical Flow로 실제 이동벡터 추정. 영상 초반(연기 전)
   프레임은 화재 미감지로 normal 종료되니, **연기 가시 구간**으로 테스트할 것.

3. **경로 라우팅**: `fire_prob<0.5 → normal 즉시종료`(flame/smoke 안 감).
   `fire_prob≥0.5 → GradCAM → HSV 화염색≥0.25면 flame, 아니면 smoke(저/고불확실)`.

## 남은 작업 (TODO)

1. **학습 완료된 가중치 연결**
   - Kaggle 커널 `yuntarwon/fireimage-training` 완료 후 Output에서 `model_save_backup.zip` 다운로드
   - 압축 풀어 `model_save/fireimage/fold0/*.pt` 배치 (git 제외 대상이라 수동 전송)
2. **상위 2개 모델 확정**
   - `results/fireimage/metrics.csv` 생기면 `model_loader.rank_top_models(2)`가 F1 상위2개 자동 선정
   - 수동 고정하려면 `model_loader.TOP_MODELS` 편집
3. (선택) 지형·연료·수원 접근성 → 지도 API/DEM 연동
4. (선택) 실시간 웹캠/CCTV: `gr.Image(sources=["webcam"])`

## 테스트 자산 (git 포함, 데이터 없이도 테스트 가능)

- `demo/test_assets/smoke_test.mp4` — 영상 입력 테스트
- `demo/test_assets/samples/*.jpg` — 화염/검은연기/흰연기/정상 샘플

## 주의: git에 없는 것

- `data/` (Kaggle 데이터셋 `yuntarwon/fireimage-*` 에 있음)
- `model_save/` 가중치 (Kaggle 학습 Output에서 받아 수동 배치)
