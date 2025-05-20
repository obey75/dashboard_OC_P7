import os

from PIL import Image
import numpy as np
import pickle




def load_models():

    import tensorflow as tf
    from tensorflow.keras.layers import Dropout, Dense
    from tensorflow.keras import Model, Input
    from tensorflow.keras import layers
    from transformers import TFViTModel, ViTConfig

    import copy

    class ViTMultiHeadClassifier(tf.keras.Model):
        def __init__(self,
                     nb_classes,
                     vit_checkpoint,
                     config_dict,
                     freeze_vit=True,
                     hidden_dim=64,
                     dropout_rate=0.5,
                     **kwargs):
            super().__init__(**kwargs)

            self.nb_classes = nb_classes
            self.vit_checkpoint = vit_checkpoint
            self.freeze_vit = freeze_vit
            self.hidden_dim = hidden_dim
            self.dropout_rate = dropout_rate

            self.config_dict = copy.deepcopy(config_dict)
            backbone_config = ViTConfig.from_dict(self.config_dict)
            self.vit = TFViTModel.from_pretrained(vit_checkpoint, config=backbone_config)
            if freeze_vit:
                self.vit.trainable = False
            self.dense_main = tf.keras.layers.Dense(hidden_dim, activation="relu")
            self.dropout_main = tf.keras.layers.Dropout(dropout_rate)
            self.out1 = tf.keras.layers.Dense(nb_classes[0], activation="sigmoid", name="output_level_1")
            self.concat = tf.keras.layers.Concatenate()
            self.dense_cond = tf.keras.layers.Dense(hidden_dim, activation="relu")
            self.dropout_cond = tf.keras.layers.Dropout(dropout_rate)
            self.out2 = tf.keras.layers.Dense(nb_classes[1], activation="sigmoid", name="output_level_2")

        def call(self, inputs, training=False):
            x = tf.transpose(inputs, perm=[0, 3, 1, 2])
            vit_outputs = self.vit(x, training=training)
            x = vit_outputs.pooler_output
            x_main = self.dense_main(x)
            x_main = self.dropout_main(x_main, training=training)
            y1 = self.out1(x_main)
            x_cond = self.concat([x, y1])
            x_cond = self.dense_cond(x_cond)
            x_cond = self.dropout_cond(x_cond, training=training)
            y2 = self.out2(x_cond)
            return {"output_level_1": y1, "output_level_2": y2}

        def get_config(self):
            return {
                "nb_classes": self.nb_classes,
                "vit_checkpoint": self.vit_checkpoint,
                "config_dict": dict(self.config_dict),  # one-level copy, safe!
                "freeze_vit": self.freeze_vit,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate
            }

        @classmethod
        def from_config(cls, config):
            return cls(**config)


    vit_model = tf.keras.models.load_model("models/ViT_model.keras",
                   custom_objects={"ViTMultiHeadClassifier": ViTMultiHeadClassifier})

    resnet_model = tf.keras.models.load_model("models/Resnet_model.keras")

    return resnet_model, vit_model


def load_mlb_objects():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/mlb_level_1.pkl"), "rb") as file:
        mlb_level_1 = pickle.load(file)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/mlb_level_2.pkl"), "rb") as file:
        mlb_level_2 = pickle.load(file)
    return mlb_level_1, mlb_level_2


def preprocess_input(_img_path):

    img = Image.open(_img_path).convert("RGB")
    img = img.resize((224, 224))

    img_np = np.array(img) / 255.0
    img_np = np.expand_dims(img_np, axis=0)

    return img_np


def predict_vit_on_img(_input, _vit_model):
    res = _vit_model.predict(_input)
    return res


def predict_resnet_on_img(_input, _resnet_model):
    res = _resnet_model.predict(_input)
    return res


def format_output_vit(_output, _mlb_level_1, _mlb_level_2):
    T1_res = {}
    T2_res = {}

    for i,score in enumerate(_output['output_level_1'][0]):
        if score >= 0.5:
            T1_res[_mlb_level_1.classes_[i]] = score

    for i,score in enumerate(_output['output_level_2'][0]):
        if score >= 0.5:
            T2_res[_mlb_level_2.classes_[i]] = score

    return T1_res, T2_res


def format_output_resnet(_output, _mlb_level_1, _mlb_level_2):
    T1_res = {}
    T2_res = {}

    for i,score in enumerate(_output[0][0]):
        if score >= 0.5:
            T1_res[_mlb_level_1.classes_[i]] = score

    for i,score in enumerate(_output[1][0]):
        if score >= 0.5:
            T2_res[_mlb_level_2.classes_[i]] = score

    return T1_res, T2_res
