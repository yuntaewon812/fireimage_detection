import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import pandas as pd
import seaborn as sns

def rgb_to_lab(image):
    """RGB 이미지를 LAB 색공간으로 변환"""
    rgb = np.array(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab

def extract_color_features(image):
    """이미지에서 L*, a*, b* 평균값 추출"""
    lab = rgb_to_lab(image)
    L_mean = np.mean(lab[:, :, 0])
    a_mean = np.mean(lab[:, :, 1])
    b_mean = np.mean(lab[:, :, 2])
    return L_mean, a_mean, b_mean

def extract_morphological_features(image):
    """이미지에서 형태학적 특징 추출"""
    gray = np.array(image.convert('L'))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0, 0, 0
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0
    
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (_, (MA, ma), _) = ellipse
        if ma > 0:
            aspect_ratio = MA / ma
        else:
            aspect_ratio = 0
    else:
        aspect_ratio = 0
    
    return area, circularity, aspect_ratio

def load_and_extract_features(folder_path, label, max_samples=500):
    """폴더에서 이미지 로드 및 특징 추출"""
    features = []
    labels = []
    
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]
    files = files[:max_samples]
    
    for filename in tqdm(files, desc=f"Processing {label}"):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            L_mean, a_mean, b_mean = extract_color_features(img)
            area, circularity, aspect_ratio = extract_morphological_features(img)
            
            features.append({
                'L_mean': L_mean,
                'a_mean': a_mean,
                'b_mean': b_mean,
                'Area': area,
                'Circularity': circularity,
                'Aspect_ratio': aspect_ratio,
                'label': label
            })
            labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return features

def statistical_comparison(df, output_dir):
    """통계적 비교 수행"""
    feature_cols = ['L_mean', 'a_mean', 'b_mean', 'Area', 'Circularity', 'Aspect_ratio']
    
    normal_data = df[df['label'] == 'normal']
    abnormal_data = df[df['label'] == 'abnormal']
    
    results = []
    for col in feature_cols:
        t_stat, p_value = stats.ttest_ind(normal_data[col], abnormal_data[col])
        results.append({
            'Feature': col,
            'Normal_mean': normal_data[col].mean(),
            'Normal_std': normal_data[col].std(),
            'Abnormal_mean': abnormal_data[col].mean(),
            'Abnormal_std': abnormal_data[col].std(),
            't_statistic': t_stat,
            'p_value': p_value
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'statistical_comparison.csv'), index=False)
    
    return results_df

def plot_violin(df, dataset_name, output_dir):
    """Violin plot 생성"""
    feature_cols = ['L_mean', 'a_mean', 'b_mean', 'Area', 'Circularity', 'Aspect_ratio']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        sns.violinplot(data=df, x='label', y=col, ax=axes[idx], palette=['#0055A4', '#D84315'])
        axes[idx].set_xlabel('', fontsize=28)
        axes[idx].set_ylabel(col, fontsize=28)
        axes[idx].tick_params(labelsize=24)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_violin.png'), dpi=300)
    plt.close()

def analyze_features(data_root, dataset_name, output_dir, max_samples=500):
    """특징 분석 수행"""
    os.makedirs(output_dir, exist_ok=True)
    
    normal_path = os.path.join(data_root, 'normal')
    abnormal_path = os.path.join(data_root, 'abnormal')
    
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name}")
    print(f"{'='*50}")
    
    normal_features = load_and_extract_features(normal_path, 'normal', max_samples)
    abnormal_features = load_and_extract_features(abnormal_path, 'abnormal', max_samples)
    
    all_features = normal_features + abnormal_features
    df = pd.DataFrame(all_features)
    
    print(f"Total samples: {len(df)} (normal: {len(normal_features)}, abnormal: {len(abnormal_features)})")
    
    stats_results = statistical_comparison(df, output_dir)
    print("\nStatistical comparison:")
    print(stats_results)
    
    plot_violin(df, dataset_name, output_dir)
    
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    base_path = r'C:\Users\308\fireimageproject\data\fireimage'
    output_base = r'C:\Users\308\fireimageproject\analysis\result'
    
    datasets = [
        ('.', 'fire_analysis')  # 'fireimage' 대신 '.'을 넣으세요.
    ]
    
    for folder_name, dataset_name in datasets:
        # base_path와 '.'이 합쳐져서 정확히 fireimage 폴더를 가리키게 됩니다.
        data_root = os.path.join(base_path, folder_name)
        output_dir = os.path.join(output_base, dataset_name)
        analyze_features(data_root, dataset_name, output_dir, max_samples=500)
    
    print("\nAll analyses completed!")
