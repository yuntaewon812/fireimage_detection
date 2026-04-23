import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# ==========================
# 설정 부분
# ==========================
class_name = "fire_analysis"  # 분석할 클래스명
base_dir = r'C:\Users\308\fireimageproject\data\fireimage'
save_dir = r'C:\Users\308\fireimageproject\analysis\result\visualization'
os.makedirs(save_dir, exist_ok=True)

# ==========================
# 이미지 로드 함수
# ==========================
# 기존 load_images_from_folder 함수를 아래 내용으로 통째로 바꾸세요.
def load_images_from_folder(folder, label, img_size=(64, 64)):
    data = []
    labels = []
    if not os.path.exists(folder):
        print(f"⚠️ 폴더를 찾을 수 없습니다: {folder}")
        return np.array([]), np.array([])

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            try:
                # 한글 경로를 읽기 위한 특수 처리 (cv2.imread 대신 사용)
                img_array = np.fromfile(path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                    
                img = cv2.resize(img, img_size)
                img_flat = img.flatten()
                data.append(img_flat)
                labels.append(label)
            except Exception as e:
                print(f"파일 로드 실패: {filename}, 에러: {e}")
                
    print(f"✅ {label} 클래스: {len(data)}장의 이미지를 로드했습니다.")
    return np.array(data), np.array(labels)

# ==========================
# 데이터 로드
# ==========================
normal_dir = os.path.join(base_dir, "normal")
abnormal_dir = os.path.join(base_dir, "abnormal")

normal_data, normal_labels = load_images_from_folder(normal_dir, "Normal")
abnormal_data, abnormal_labels = load_images_from_folder(abnormal_dir, "Abnormal")

X = np.vstack((normal_data, abnormal_data))
y = np.concatenate((normal_labels, abnormal_labels))

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# t-SNE 시각화
# ==========================
# n_iter 옵션을 제거합니다. 기본값으로도 충분히 잘 작동합니다.
tsne = TSNE(n_components=2, random_state=1004, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for label, color in zip(["Normal", "Abnormal"], ["blue", "red"]):
    plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=label, alpha=0.6, s=20, color=color)

plt.title(f"t-SNE Visualization ({class_name})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{class_name}_tsne.png"), dpi=300)
plt.close()

# ==========================
# UMAP 시각화
# ==========================
reducer = umap.UMAP(n_components=2, random_state=1004, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for label, color in zip(["Normal", "Abnormal"], ["blue", "red"]):
    plt.scatter(X_umap[y == label, 0], X_umap[y == label, 1], label=label, alpha=0.6, s=20, color=color)

plt.title(f"UMAP Visualization ({class_name})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{class_name}_umap.png"), dpi=300)
plt.close()

print(f"✅ t-SNE, UMAP 시각화 결과가 저장되었습니다: {save_dir}")

