import os
import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import datetime
from zoneinfo import ZoneInfo

from utils.utils import save_csv, plot_confusion_matrix, roc_plot


# =========================
# Online 증강 (배치 텐서 → 텐서)
# =========================
def augment_batch(x: torch.Tensor, augment: bool = False) -> torch.Tensor:
    """
    GPU 벡터화 증강 — Python for 루프 없이 배치 전체를 텐서 연산으로 처리.
    CPU 병목 제거 → GPU 효율 최대화.
    """
    if not augment:
        return x

    device = x.device
    B, C, H, W = x.shape
    x = x.clone()

    # 1. RandomHorizontalFlip (GPU, 배치 전체)
    flip = torch.rand(B, device=device) < 0.5            # (B,) bool
    x[flip] = x[flip].flip(-1)

    # 2. ColorJitter (GPU, 배치 공통 계수 — 빠른 도메인 랜덤화)
    if random.random() < 0.8:
        b = 1.0 + random.uniform(-0.3, 0.3)
        c = 1.0 + random.uniform(-0.3, 0.3)
        s = 1.0 + random.uniform(-0.3, 0.3)
        x = torch.clamp(x * b, 0.0, 1.0)
        m = x.mean(dim=[-2, -1], keepdim=True)
        x = torch.clamp(m + (x - m) * c, 0.0, 1.0)
        # 채도: RGB → 명도 채널 혼합
        gray = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
        x = torch.clamp(gray + (x - gray) * s, 0.0, 1.0)

    # 3. Random Erasing (GPU — 좌표만 CPU, 마스크는 GPU)
    er = torch.rand(B, device=device) < 0.5
    if er.any():
        rh = max(1, int(H * random.uniform(0.1, 0.3)))
        rw = max(1, int(W * random.uniform(0.1, 0.3)))
        y0 = random.randint(0, H - rh)
        x0 = random.randint(0, W - rw)
        x[er, :, y0:y0+rh, x0:x0+rw] = 0.0

    # 4. RandomVerticalFlip (보조)
    vflip = torch.rand(B, device=device) < 0.2
    x[vflip] = x[vflip].flip(-2)

    return x


def augment_batch_youtube(x: torch.Tensor) -> torch.Tensor:
    """
    YouTube 도메인 특화 증강 (설정 E).

    CCTV 학습 데이터와 YouTube 실환경 영상의 시각적 도메인 갭을 줄이기 위해
    YouTube 고유의 열화(degradation) 패턴을 학습 시 미리 경험시킨다.

    표준 증강(C)과 누적 적용:
      1. JPEG/영상 압축 아티팩트 시뮬레이션
      2. 모션 블러 (핸드헬드·헬기 카메라)
      3. 해상도 저하 (저화질 업로드 후 업스케일)
      4. 색온도 시프트 (카메라별 화이트밸런스 차이)
      5. 자막/로고 오버레이 시뮬레이션 (하단 밴드 마스킹)
    """
    import torch.nn.functional as F

    device = x.device
    B, C, H, W = x.shape
    x = x.clone()

    # ── 1. JPEG 압축 아티팩트 시뮬레이션 ─────────────────────────
    # 고주파 노이즈 + 약한 평균 블러 → 압축 블록 노이즈와 유사한 효과
    if random.random() < 0.6:
        noise_std = random.uniform(0.01, 0.06)
        noise = torch.randn_like(x) * noise_std
        x = torch.clamp(x + noise, 0.0, 1.0)
        # 3×3 평균 블러로 블록 경계 뭉개기
        kernel = torch.ones(C, 1, 3, 3, device=device) / 9.0
        x = F.conv2d(x, kernel, padding=1, groups=C)
        x = torch.clamp(x, 0.0, 1.0)

    # ── 2. 모션 블러 ──────────────────────────────────────────────
    # 수평/수직 방향성 블러 커널로 카메라 움직임 시뮬레이션
    if random.random() < 0.4:
        k = random.choice([5, 7, 9])
        if random.random() < 0.5:
            # 수평 블러
            kernel = torch.zeros(C, 1, 1, k, device=device)
            kernel[:, :, 0, :] = 1.0 / k
        else:
            # 수직 블러
            kernel = torch.zeros(C, 1, k, 1, device=device)
            kernel[:, :, :, 0] = 1.0 / k
        pad = k // 2
        x = F.conv2d(x, kernel, padding=(kernel.shape[2]//2, kernel.shape[3]//2), groups=C)
        x = torch.clamp(x, 0.0, 1.0)

    # ── 3. 해상도 저하 (다운샘플 → 업샘플) ───────────────────────
    # 저화질 영상 업로드 시 생기는 픽셀 뭉개짐 재현
    if random.random() < 0.5:
        scale = random.uniform(0.4, 0.75)
        h_small = max(16, int(H * scale))
        w_small = max(16, int(W * scale))
        x = F.interpolate(x, size=(h_small, w_small), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = torch.clamp(x, 0.0, 1.0)

    # ── 4. 색온도 시프트 ──────────────────────────────────────────
    # 뉴스 카메라 vs CCTV 화이트밸런스 차이 (파랑↔노랑 축)
    if random.random() < 0.5:
        shift = random.uniform(-0.08, 0.08)
        x[:, 0] = torch.clamp(x[:, 0] + shift, 0.0, 1.0)   # R 채널
        x[:, 2] = torch.clamp(x[:, 2] - shift, 0.0, 1.0)   # B 채널

    # ── 5. 자막/로고 오버레이 시뮬레이션 ─────────────────────────
    # 뉴스 자막이 주로 위치하는 하단 영역을 반투명 검정으로 마스킹
    if random.random() < 0.3:
        band_h = int(H * random.uniform(0.06, 0.15))
        y_start = int(H * random.uniform(0.75, 0.88))
        y_end = min(H, y_start + band_h)
        alpha = random.uniform(0.3, 0.7)   # 불투명도
        x[:, :, y_start:y_end, :] *= (1.0 - alpha)

    return x


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Mixup: 두 샘플을 가중 혼합해 경계 부드럽게 → OOD 일반화↑"""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam


# =========================
# ECE (Expected Calibration Error)
# =========================
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    모델 신뢰도(confidence)와 실제 정확도(accuracy)의 불일치를 측정.
    0에 가까울수록 잘 보정된 모델. 일반적으로 0.05 이하면 양호.

    Args:
        y_true: 실제 레이블 (0 or 1)
        y_prob: 클래스 1(화재)에 대한 예측 확률
        n_bins: 신뢰도 구간 개수
    Returns:
        ECE 값 [0, 1]
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        acc_in_bin = float((y_true[mask] == 1).mean())
        conf_in_bin = float(y_prob[mask].mean())
        ece += abs(acc_in_bin - conf_in_bin) * (n_in_bin / n)
    return ece

# =========================
# 1️⃣ 한 epoch 학습
# =========================
def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    augment: bool = False, use_mixup: bool = False,
                    augment_type: str = 'standard'):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Online 증강
        X_batch = augment_batch(X_batch, augment=augment)
        if augment and augment_type == 'youtube':
            X_batch = augment_batch_youtube(X_batch)

        optimizer.zero_grad()
        if use_mixup:
            X_mix, y_a, y_b, lam = mixup_batch(X_batch, y_batch)
            outputs = model(X_mix)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_samples += y_batch.size(0)

    return total_loss / total_samples, total_correct / total_samples


# =========================
# 2️⃣ 평가 함수 (val/test 공용)
# =========================
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_labels, all_preds, all_probs, all_losses = [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1] if outputs.size(1) > 1 else torch.sigmoid(outputs)

            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_losses.extend([loss.item()] * X_batch.size(0))  # batch loss 기록

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array(all_losses)


# =========================
# 3️⃣ Bootstrap CI 계산
# =========================
def bootstrap_ci(metric_fn, y_true, y_pred, probs=None, n_bootstrap=1000, alpha=0.95):
    rng = np.random.default_rng()
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_true_sample = y_true[idx]
        y_pred_sample = y_pred[idx]
        if probs is not None:
            probs_sample = probs[idx]
            scores.append(metric_fn(y_true_sample, probs_sample))
        else:
            scores.append(metric_fn(y_true_sample, y_pred_sample))

    lower = np.percentile(scores, ((1 - alpha) / 2) * 100)
    upper = np.percentile(scores, (1 - (1 - alpha) / 2) * 100)
    return [np.mean(scores), lower, upper]


def bootstrap_ci_loss(values, n_bootstrap=1000, alpha=0.95):
    rng = np.random.default_rng()
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sample = values[idx]
        means.append(sample.mean())
    lower = np.percentile(means, ((1 - alpha) / 2) * 100)
    upper = np.percentile(means, (1 - (1 - alpha) / 2) * 100)
    return [np.mean(means), lower, upper]


# =========================
# 4️⃣ 전체 학습/평가 루프
# =========================
def train_full_loop(train_loader, val_loader, test_loader, model_dicts, class_name, fold_num,
                    lr=1e-3, epochs=50, patience=10, device='cuda',
                    per_model_lr: dict = None,
                    force_retrain: set = None,
                    use_scheduler: set = None,
                    augment: bool = False,
                    use_mixup: bool = False,
                    augment_type: str = 'standard'):

    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        _xpu = torch.xpu.is_available()
    except ImportError:
        _xpu = False
    if _xpu:
        device = torch.device('xpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        cap = torch.cuda.get_device_capability()
        print(f"  GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    else:
        device = torch.device('cpu')
    print(f"  device: {device}")
    per_model_lr   = per_model_lr  or {}
    force_retrain  = force_retrain or set()
    # 기본 스케줄러 적용 대상: 대형 transformer 계열
    default_scheduler_models = {'internimage', 'maxvit', 'nextvit', 'swintransformerv2', 'convnextv2'}
    use_scheduler = use_scheduler if use_scheduler is not None else default_scheduler_models

    for model_name, model in model_dicts.items():
        print(f"\n=== Training {model_name} on Fold {fold_num} ===")

        save_path = f'./model_save/{class_name}/fold{fold_num}/{model_name}.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # force_retrain 대상이면 기존 가중치 삭제 후 재학습
        if model_name in force_retrain and os.path.exists(save_path):
            os.remove(save_path)
            print(f"  [force_retrain] 기존 가중치 삭제 → 재학습")

        model.to(device)
        criterion = nn.CrossEntropyLoss()

        # 모델별 학습률 (per_model_lr 우선, 없으면 lr 기본값)
        model_lr = per_model_lr.get(model_name, lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_lr, weight_decay=1e-4)
        print(f"  lr={model_lr:.1e}  weight_decay=1e-4")

        # Cosine Annealing 스케줄러 (대형 transformer 계열에 적용)
        scheduler = None
        if model_name in use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=model_lr * 0.01)
            print(f"  CosineAnnealingLR 적용 (T_max={epochs})")

        best_val_acc = 0
        patience_counter = 0

        try:
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            print(f"  기존 가중치 로드 완료 → 평가만 수행")
            training_needed = False
        except Exception as e:
            print(f"  저장된 가중치 없음 → 처음부터 학습: {e}")
            training_needed = True

        best_val_acc = 0
        patience_counter = 0

        # -----------------------------
        # 학습
        # -----------------------------
        if training_needed:
            for epoch in range(epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                                           augment=augment, use_mixup=use_mixup,
                                                           augment_type=augment_type)
                val_loss, val_acc, _, _, _, _ = evaluate_model(model, val_loader, criterion, device)

                if scheduler is not None:
                    scheduler.step()

                log_line = (f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                print(log_line)

                try:
                    with open('./training_progress.txt', 'a', encoding='utf-8') as pf:
                        pf.write(f"[{model_name} fold{fold_num}] {log_line}\n")
                except Exception:
                    pass

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # -----------------------------
        # 테스트 평가
        # -----------------------------
        model.load_state_dict(torch.load(save_path, map_location=device))
        test_loss, test_acc, y_true, y_pred, y_prob, all_losses = evaluate_model(model, test_loader, criterion, device)

        # Bootstrap metrics
        test_loss_ci = bootstrap_ci_loss(all_losses)
        test_acc_ci = bootstrap_ci(balanced_accuracy_score, y_true, y_pred)
        test_recall_ci = bootstrap_ci(lambda y, yhat: recall_score(y, yhat, average='macro'), y_true, y_pred)
        test_prec_ci = bootstrap_ci(lambda y, yhat: precision_score(y, yhat, average='macro'), y_true, y_pred)
        test_f1_ci = bootstrap_ci(lambda y, yhat: f1_score(y, yhat, average='macro'), y_true, y_pred)

        try:
            test_auc_ci = bootstrap_ci(lambda y, p: roc_auc_score(y, p, multi_class='ovr'), y_true, y_pred, probs=y_prob)
        except Exception as e:
            print(f"ROC AUC calculation failed: {e}")
            test_auc_ci = [np.nan, np.nan, np.nan]

        # ECE (보정 오차)
        ece = compute_ece(y_true, y_prob)
        ece_status = "양호" if ece <= 0.05 else ("주의" if ece <= 0.10 else "불량")
        print(f"  ECE: {ece:.4f}  [{ece_status}]  "
              f"(0.05 이하=양호 / F1={test_f1_ci[0]:.3f} / Loss={test_loss_ci[0]:.3f})")

        # -----------------------------
        # 결과 저장
        # -----------------------------
        now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
        save_csv(f'{model_name}_{fold_num}', test_acc_ci, test_loss_ci, test_recall_ci,
                 test_prec_ci, test_f1_ci, test_auc_ci, class_name, now, ece=ece)

        plot_confusion_matrix(y_true, y_pred, model_name, class_name, fold_num)
        #roc_plot(y_true, y_prob, model_name, class_name, fold_num)
