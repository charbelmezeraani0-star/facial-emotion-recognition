import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation attention
# ---------------------------------------------------------------------------

def se_block(x, ratio=16):
    """Channel attention via SE block — simpler and more gradient-friendly than CBAM."""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(filters // ratio, 4), activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def conv_bn_relu(x, filters, kernel_size=3, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)


def residual_block(x, filters):
    """Pre-activation residual block with SE attention."""
    in_filters = x.shape[-1]
    shortcut = x
    if in_filters != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                                 kernel_initializer='he_normal')(x)
        shortcut = layers.BatchNormalization()(shortcut)

    x = conv_bn_relu(x, filters)
    x = conv_bn_relu(x, filters)
    x = se_block(x)
    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)


# ---------------------------------------------------------------------------
# Main model — Multi-Scale Feature Fusion CNN with SE Attention
# ---------------------------------------------------------------------------

def build_custom_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Multi-scale feature fusion CNN with Squeeze-and-Excitation attention.
    Extracts features at 4 spatial resolutions and fuses them for classification.
    """
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = conv_bn_relu(inputs, 64)           # 48×48×64

    # Scale 1 — fine-grained local features
    s1 = residual_block(x, 64)
    s1 = layers.MaxPooling2D(2)(s1)        # 24×24×64
    s1 = layers.Dropout(0.2)(s1)

    # Scale 2
    s2 = residual_block(s1, 128)
    s2 = layers.MaxPooling2D(2)(s2)        # 12×12×128
    s2 = layers.Dropout(0.2)(s2)

    # Scale 3
    s3 = residual_block(s2, 256)
    s3 = layers.MaxPooling2D(2)(s3)        # 6×6×256
    s3 = layers.Dropout(0.2)(s3)

    # Scale 4 — global context
    s4 = residual_block(s3, 512)           # 6×6×512

    # Multi-scale pooling: pool each scale independently, project to same dim, then fuse
    p1 = layers.GlobalAveragePooling2D()(s1)   # 64-d
    p2 = layers.GlobalAveragePooling2D()(s2)   # 128-d
    p3 = layers.GlobalAveragePooling2D()(s3)   # 256-d
    p4 = layers.GlobalAveragePooling2D()(s4)   # 512-d

    p1 = layers.Dense(128, use_bias=False, kernel_initializer='he_normal')(p1)
    p2 = layers.Dense(128, use_bias=False, kernel_initializer='he_normal')(p2)
    p3 = layers.Dense(128, use_bias=False, kernel_initializer='he_normal')(p3)
    p4 = layers.Dense(128, use_bias=False, kernel_initializer='he_normal')(p4)

    fused = layers.Concatenate()([p1, p2, p3, p4])   # 512-d
    fused = layers.BatchNormalization()(fused)
    fused = layers.Activation('relu')(fused)
    fused = layers.Dropout(0.4)(fused)

    fused = layers.Dense(256, use_bias=False, kernel_initializer='he_normal')(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Activation('relu')(fused)
    fused = layers.Dropout(0.3)(fused)

    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(fused)

    return Model(inputs, outputs, name='MultiScale_SE_CNN')


if __name__ == '__main__':
    build_custom_cnn().summary()
