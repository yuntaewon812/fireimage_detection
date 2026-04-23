from tensorflow.keras import layers # type: ignore[import]
from tensorflow.keras.models import Model # type: ignore[import] 
from tensorflow import keras  
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, ReLU, LeakyReLU # type: ignore[import]
from models.module import CBAMBlock, SwinTransformerBlock, convnext_block 

##################################
#          Deep model            # 
##################################

def build_model(model_name, num_classes=2, input_shape=(224, 224, 3)): 
    input_tensor = layers.Input(shape=input_shape, name='image_input')

    model_dict = {
        'VGG16': keras.applications.VGG16,
        'ResNet50V2': keras.applications.ResNet50V2,
        'DenseNet121': keras.applications.DenseNet121
    }

    if model_name in model_dict:
        base_model = model_dict[model_name](include_top=False, weights=None, input_tensor=input_tensor)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    x = base_model.output
    x = CBAMBlock()(x)
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

import convnextv2 as convnextv2

def build_convnext_v2():
     model = convnextv2.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        head_init_scale=args.head_init_scale,
    )
    return Model
    
##################################
#           ConvNext             #
##################################

def build_convnext(num_classes, kernel_size=5, drop_path_rate=0.1, normalization='layernorm', input_shape=(224, 224, 3)): 
    depths=[3, 3, 9, 3]
    dims=[96, 192, 384, 768]

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Stem layer: Downsample by 4x
    x = layers.Conv2D(dims[0], kernel_size=kernel_size, strides=4, padding="same")(x)

    if normalization == "layernorm":
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    elif normalization == "batchnorm":
        x = layers.BatchNormalization()(x)

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
            if normalization == "layernorm":
                x = layers.LayerNormalization(epsilon=1e-6)(x)
            elif normalization == "batchnorm":
                x = layers.BatchNormalization()(x)
            x = layers.Conv2D(dims[stage + 1], kernel_size=kernel_size, strides=2, padding="same")(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    if normalization == "layernorm":
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    elif normalization == "batchnorm":
        x = layers.BatchNormalization()(x)
    
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else :
        outputs = layers.Dense(num_classes, activation="softmax")(x)
 
    return Model(inputs=inputs, outputs=outputs)
