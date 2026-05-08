import tensorflow as tf
from tensorflow.keras import layers, Model


def _build_transfer_model(backbone_fn, input_shape, upsample_size,
                           backbone_input_shape, num_classes, name):
    """
    Builds a transfer-learning model that accepts 48×48 grayscale input.
    Strategy: build the backbone independently with its native input shape,
    then wrap it in a larger model that handles grayscale→RGB conversion.
    This avoids the layer-count mismatch that occurs with input_tensor=.
    """
    # 1. Build backbone with its native RGB input shape (no custom input tensor)
    backbone = backbone_fn(
        input_shape=backbone_input_shape,
        include_top=False,
        weights='imagenet')
    backbone.trainable = False

    # 2. Build the wrapper model
    inp = layers.Input(shape=input_shape, name='grayscale_input')

    # Grayscale (1ch) → RGB (3ch) by triplication — gives backbone real face info
    x = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1), name='gray_to_rgb')(inp)
    # Upsample to backbone's expected resolution
    x = layers.UpSampling2D(size=upsample_size, interpolation='bilinear', name='upsample')(x)
    # Normalise to ImageNet range that the backbone expects
    x = layers.Lambda(lambda t: (t - 0.5) * 2.0, name='imagenet_norm')(x)

    # Run through backbone
    x = backbone(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inp, out, name=name)
    return model, backbone


def build_mobilenetv2(input_shape=(48, 48, 1), num_classes=7):
    return _build_transfer_model(
        backbone_fn=tf.keras.applications.MobileNetV2,
        input_shape=input_shape,
        upsample_size=(4, 4),           # 48 → 192
        backbone_input_shape=(192, 192, 3),
        num_classes=num_classes,
        name='MobileNetV2_FER')


def build_resnet50(input_shape=(48, 48, 1), num_classes=7):
    return _build_transfer_model(
        backbone_fn=tf.keras.applications.ResNet50,
        input_shape=input_shape,
        upsample_size=(4, 4),           # 48 → 192
        backbone_input_shape=(192, 192, 3),
        num_classes=num_classes,
        name='ResNet50_FER')


def build_efficientnetb2(input_shape=(48, 48, 1), num_classes=7):
    """
    EfficientNetB2 native input: 260×260.
    We upsample 48 → 240 (close enough; EfficientNet handles size variation).
    Best accuracy/efficiency ratio among the three backbones.
    """
    return _build_transfer_model(
        backbone_fn=tf.keras.applications.EfficientNetB2,
        input_shape=input_shape,
        upsample_size=(5, 5),           # 48 → 240
        backbone_input_shape=(240, 240, 3),
        num_classes=num_classes,
        name='EfficientNetB2_FER')


def unfreeze_top_layers(base_model, num_layers=30):
    """Unfreeze last N layers of backbone for fine-tuning."""
    for layer in base_model.layers[-num_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
