import os
from torch.utils.data import DataLoader, TensorDataset

from utils.utils import load_data, get_args
from utils.train_loop import train_full_loop
from models.deep_model import DeepModel, DeepSAFEModel
from models.swin_transformer import SwinTransformer, SwinTransformerSAFE
 
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import KFold

# from models.mambavision import MambaVisionForImageClassification, MambaVisionForImageClassification_v2
from models.nextvit import NextViTForImageClassification, NextViTForImageClassification_v2
from models.swintransformerv2 import SwinTransformerV2ForImageClassification, SwinTransformerV2ForImageClassification_v2
from models.convnextv2 import ConvNeXtV2ForImageClassification, ConvNeXtV2ForImageClassification_v2
from models.efficientnetv2 import EfficientNetV2ForImageClassification, EfficientNetV2ForImageClassification_v2
from models.gpt_vit import GPT2ForImageClassification
from models.bert_vit import BertForImageClassification,BertForImageClassification_v2

import datetime
from zoneinfo import ZoneInfo  


os.makedirs('./model_save', exist_ok=True)

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print('='*30)
print(now.strftime('%Y-%m-%d %H:%M:%S'))
print('='*30)
 
args = get_args()
class_name = args.class_name
print(f'now training for class: {class_name}')

X, y = load_data(class_name)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1004)

for i, (train_index, test_index) in enumerate(skf.split(X,y)):
    train_idx, val_idx = train_test_split(train_index, test_size = 0.25, random_state = 1004)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_index], y[test_index]



    train_dataset1 = TensorDataset(X_train, y_train)
    val_dataset1 = TensorDataset(X_val, y_val)
    test_dataset1 = TensorDataset(X_test, y_test)
 
    train_loader1 = DataLoader(train_dataset1, batch_size=8, shuffle=True)
    val_loader1 = DataLoader(val_dataset1, batch_size=8, shuffle=False)
    test_loader1 = DataLoader(test_dataset1, batch_size=8, shuffle=False)

    model_dicts = {
        # 'mambavision_proposal': MambaVisionForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny'),
        'efficientnetv2_proposal': EfficientNetV2ForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='s'),
        # 'mambavision': MambaVisionForImageClassification(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny'),
        'nextvit': NextViTForImageClassification(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='small'),
        'efficientnetv2': EfficientNetV2ForImageClassification(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='s'),
        'Resnet50': DeepModel('ResNet50'),
        'DenseNet121': DeepModel('DenseNet121'),
    }
    train_full_loop(train_loader1, val_loader1, test_loader1, model_dicts, class_name, i)

print('finished')
