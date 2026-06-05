import sys, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
x = torch.randn(2, 3, 160, 160)

def test(name, fn):
    try:
        m = fn(); m.eval()
        out = m(x)
        print(f'OK  {name:30s} out={out.shape}')
    except Exception as e:
        print(f'NG  {name:30s} {str(e)[:90]}')

from models.deep_model import DeepModel
test('ResNet50',    lambda: DeepModel('ResNet50', pretrained=True))
test('DenseNet121', lambda: DeepModel('DenseNet121', pretrained=True))

from models.efficientnetv2 import EfficientNetV2ForImageClassification as E1
from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2 as E2
test('efficientnetv2',
     lambda: E1(num_labels=2,img_size=160,patch_size=16,hidden_dim=512,model_variant='s',pretrained=True))
test('efficientnetv2_proposal',
     lambda: E2(num_labels=2,img_size=160,patch_size=16,hidden_dim=512,model_variant='s',pretrained=True))

from models.nextvit import NextViTForImageClassification
test('nextvit',
     lambda: NextViTForImageClassification(num_labels=2,img_size=160,patch_size=16,hidden_dim=512,model_variant='small',pretrained=True))

from models.maxvit import MaxViTForImageClassification
test('maxvit',
     lambda: MaxViTForImageClassification(num_labels=2,img_size=160,patch_size=16,hidden_dim=512,model_variant='tiny',pretrained=True))

from models.internimage import InternImageForImageClassification
test('internimage(Swin-Tiny)',
     lambda: InternImageForImageClassification(num_labels=2,img_size=160,patch_size=16,hidden_dim=512,model_variant='tiny',pretrained=True))

print('\n=== 완료 ===')
