import os
import cv2
import torch
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc 
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description='class 입력') 
    parser.add_argument('--class_name', type=str, default='abnormal', help='클래스 이름을 입력하세요')
    args = parser.parse_args()
    return args

def load_data(class_name, img_size=(256, 256), device='cpu'):
    X, y = [], []
    # 팁: 경로 구분자를 명확히 하기 위해 os.path.join 권장
    data_path = os.path.join('.', 'data', class_name)

    # fireimage 클래스의 경우 하위 폴더가 'normal', 'fire'일 수 있으므로 유연하게 대처
    target_subfolders = ['normal', 'abnormal', 'fire'] 
    
    for cls_name in target_subfolders:
        cls_path = os.path.join(data_path, cls_name)
        if not os.path.exists(cls_path):
            continue
            
        label = 0 if cls_name == 'normal' else 1

        for file_name in os.listdir(cls_path):
            image_path = os.path.join(cls_path, file_name)
            
            # [수정] 한글 경로 및 특수문자 포함 파일 로딩 문제 해결
            try:
                img_array = np.fromfile(image_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    X.append(img)
                    y.append(label)
                else:
                    print(f"경고: 파일을 읽을 수 없습니다 -> {image_path}")
            except Exception as e:
                print(f"에러 발생 ({file_name}): {e}")
 
    if not X:
        raise ValueError(f"데이터를 찾을 수 없습니다. 경로를 확인하세요: {data_path}")

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int64)
 
    X = np.transpose(X, (0, 3, 1, 2)) # [N, H, W, C] -> [N, C, H, W]
 
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    return X_tensor, y_tensor

def save_csv(model_name, acc, loss, recall, prec, f1, auc_val, class_name, time_val) : 
    # 결과 포맷팅 (평균, 최소, 최대)
    acc_str = f'{acc[0]:.3f}({acc[1]:.3f}-{acc[2]:.3f})'
    loss_str = f'{loss[0]:.3f}({loss[1]:.3f}-{loss[2]:.3f})'
    recall_str = f'{recall[0]:.3f}({recall[1]:.3f}-{recall[2]:.3f})'
    prec_str = f'{prec[0]:.3f}({prec[1]:.3f}-{prec[2]:.3f})'
    f1_str = f'{f1[0]:.3f}({f1[1]:.3f}-{f1[2]:.3f})'
    
    auc_str = f'{auc_val[0]:.3f}({auc_val[1]:.3f}-{auc_val[2]:.3f})' if auc_val is not None else 'Nan(Nan-Nan)'
    time_str = time_val.strftime("%Y-%m-%d %H:%M:%S")

    output_dir = os.path.join('.', 'results', class_name)
    os.makedirs(output_dir, exist_ok=True)

    header = ['model name', 'Accuracy', 'Loss', 'Recall', 'Precision', 'F1 score', 'AUROC', 'timestamp']
    new_row = [model_name, acc_str, loss_str, recall_str, prec_str, f1_str, auc_str, time_str]

    csv_filename = os.path.join(output_dir, 'metrics.csv')

    if not os.path.exists(csv_filename):
        df = pd.DataFrame([new_row], columns=header)
    else:
        df = pd.read_csv(csv_filename)
        if model_name in df['model name'].values:
            df.loc[df['model name'] == model_name, header[1:]] = new_row[1:]
        else:
            df = pd.concat([df, pd.DataFrame([new_row], columns=header)], ignore_index=True)
    
    df.to_csv(csv_filename, index=False, header=True) 

def plot_confusion_matrix(y_test, y_pred, model_name, class_name, fold_num) :
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = y_pred 
    
    class_names = ['Normal', class_name] 
    confusion = confusion_matrix(y_test, y_pred_labels)
    cm_df = pd.DataFrame(confusion, index=class_names, columns=class_names)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, annot=True, cmap='Purples', fmt='d', cbar=False, annot_kws={"size": 60})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name} (Fold {fold_num})')

    save_path = os.path.join('.', 'results', class_name, 'cm')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{model_name}_{fold_num}.png"))
    plt.close()

# [수정] TensorFlow 의존성 제거를 위한 직접 구현 (one-hot encoding)
def my_to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

def roc_plot(y_test, y_prob, model_name, class_name, fold_num):
    # y_prob가 1차원(확률값 하나)일 경우 2차원 클래스 확률로 변환
    if y_prob.ndim == 1:
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)
 
    # 자체 구현한 원-핫 인코딩 함수 사용
    y_test_oh = my_to_categorical(y_test, num_classes=2)

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    class_labels = ['Normal', class_name]
 
    for i in range(2):
        fpr, tpr, _ = roc_curve(y_test_oh[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_labels[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')

    save_path = os.path.join('.', 'results', class_name, 'roc')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{model_name}_{fold_num}.png"))
    plt.close()