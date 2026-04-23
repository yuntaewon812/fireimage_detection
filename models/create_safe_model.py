from tensorflow import keras
from tensorflow.keras.models import Model # type: ignore[import] 
from tensorflow.keras import layers, Model # type: ignore[import]
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, ReLU, LeakyReLU # type: ignore[import]
from models.module import CBAMBlock,  SwinTransformerBlock, convnext_block, safe_module 

##################################
#           Deep model           #
##################################

def build_model_safe(model_name, num_classes=2, input_shape=(224, 224, 3)):
    input_tensor = layers.Input(shape=input_shape, name='image_input')

    model_dict = {
        'VGG16': keras.applications.VGG16,
        'ResNet50V2': keras.applications.ResNet50V2,
        'DenseNet121': keras.applications.DenseNet121
        "ConvNextTiny": keras.applications.ConvNeXtTiny
        "EfficientNetV2BO":keras.applications.EfficientNetV2B0
    }

    if model_name in model_dict:
        base_model = model_dict[model_name](include_top=False, weights=None, input_tensor=input_tensor)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    x = base_model.output
    x = CBAMBlock()(x)
    x = safe_module(x) 
    x = GlobalAveragePooling2D()(x)

    for i in range(2):
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x) if i == 0 else ReLU()(x)   
        x = Dropout(0.5)(x) 

    combined = Dense(256)(x)
    combined = BatchNormalization()(combined)
    combined = ReLU()(combined)
    combined = Dropout(0.5)(combined)

    if num_classes == 1:
        output = Dense(1, activation='sigmoid')(combined)
    else:
        output = Dense(num_classes, activation='softmax')(combined)

    return Model(inputs=input_tensor, outputs=output)

##################################
#       swin transformer         #
##################################

def build_swin_safe(num_classes, window_size=4, mlp_ratio=2.0, num_heads=2, input_shape=(224, 224, 3)):
    embed_dim = 32
    num_blocks=2

    inputs = layers.Input(shape=input_shape)

    # 패치 임베딩
    x = layers.Conv2D(embed_dim, kernel_size=4, strides=4, padding="same")(inputs)
    x = layers.LayerNormalization()(x)
    x = safe_module(x)
    x = layers.Conv2D(embed_dim, kernel_size=1, padding='same')(x)

    H, W = input_shape[0] // 4, input_shape[1] // 4

    # Swin Transformer 블록 추가 (경량화 적용)
    for i in range(num_blocks):
        shift_size = 0 if i % 2 == 0 else window_size // 2
        x = SwinTransformerBlock(embed_dim, num_heads, window_size, mlp_ratio, shift_size)(x, H=H, W=W)

    # 분류기 헤드
    x = layers.GlobalAveragePooling2D()(x)
    
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else :
        outputs = layers.Dense(num_classes, activation="softmax")(x)
 
    return Model(inputs=inputs, outputs=outputs)

##################################
#           ConvNext             #
##################################

def build_convnext_safe(num_classes, kernel_size=5, drop_path_rate=0.1, input_shape=(224, 224, 3)): 
    depths=[3, 3, 9, 3]
    dims=[96, 192, 384, 768]

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Stem layer: Downsample by 4x
    x = layers.Conv2D(dims[0], kernel_size=kernel_size, strides=4, padding="same")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = safe_module(x)
    
    # Stages
    total_blocks = sum(depths)
    block_idx = 0
    for stage, (depth, dim) in enumerate(zip(depths, dims)):
        for i in range(depth):
            drop_rate = drop_path_rate * block_idx / total_blocks
            x = convnext_block(x, dim, drop_path_rate=drop_rate)
            block_idx += 1

        # Downsample between stages
        if stage < len(depths) - 1:
            x = layers.LayerNormalization(epsilon=1e-6)(x) 
            x = layers.Conv2D(dims[stage + 1], kernel_size=kernel_size, strides=2, padding="same")(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else :
        outputs = layers.Dense(num_classes, activation="softmax")(x)
 
    return Model(inputs=inputs, outputs=outputs)
