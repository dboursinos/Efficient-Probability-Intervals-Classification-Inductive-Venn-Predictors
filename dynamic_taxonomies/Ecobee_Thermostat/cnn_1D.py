from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Conv1D, MaxPool1D, Dropout, Flatten, Dense, Activation, Add, Input, AveragePooling1D, Concatenate, Lambda
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras import backend as K
from keras.optimizers import SGD,Adam

def residual_block(x, filters, conv_num=3, activation="relu"):
    s = Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding="same")(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding="same")(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPool1D(pool_size=2, strides=2)(x)

def create_base_model(input_shape, embeddings,num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = AveragePooling1D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(units= embeddings, activation="relu",name='embedding')(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    base_model=Model(inputs, outputs)
    base_model.compile(loss="sparse_categorical_crossentropy",
                 optimizer="Adam",
                 metrics=['accuracy'])
    return base_model
