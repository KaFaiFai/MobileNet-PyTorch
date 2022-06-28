"""
convert from
- MobileNetTF: official tensorflow pretrained model
- MobileNet: wjc852456's repo https://github.com/wjc852456/pytorch-mobilenet-v1
- MobileNetV2: official pytorch pretrained model
"""
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2
import tensorflow as tf
import numpy as np

from model import *
from converter import *





def peek_pytorch_state():
    network = MobileNetV2(1000)
    # network = mobilenet_v2(pretrained=True)
    network_state = network.state_dict()

    # pretrained_model_path = r"C:\_Project\Pycharm Projects\MobileNet\pretrained\mobilenet_sgd_68.848.pth.tar"
    # state = torch.load(pretrained_model_path)
    # network.load_state_dict(state["state_dict"])
    # network_state = state["state_dict"]

    print("PyTorch model's state_dict:")
    for param_tensor in network_state:
        dims = len(network_state[param_tensor].size())
        if dims == 1:
            list_peek = network_state[param_tensor][:5].tolist()
        elif dims == 4:
            list_peek = network_state[param_tensor][:5, :5, :5, :5].tolist()
        else:
            list_peek = []
        print(f"{param_tensor:<60}: {network_state[param_tensor].size()}")
        # f" {list_peek}")


def peek_tensorflow_state():
    model = tf.keras.applications.mobilenet.MobileNet(
        input_shape=(224, 224, 3),
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
    )
    # model.summary()
    # inputs = tf.keras.Input(shape=(32, 32, 3))
    # x = tf.keras.layers.Conv2D(5, (2, 3))(inputs)
    # outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)

    random_input = np.random.randn(5, 224, 224, 3)
    # x = tf.convert_to_tensor(random_input, dtype=tf.float32)
    x = tf.random.normal((5, 224, 224, 3))
    # with tf.GradientTape() as tape:
    predictions = model(x)
    print(predictions)

    # for layer in model.layers:
    #     if len(layer.get_weights()) > 0:
    #         for t, w in zip(layer.weights, layer.get_weights()):
    #             dims = len(w.shape)
    #             if dims == 1:
    #                 list_peek = w[:5].tolist()
    #             elif dims == 4:
    #                 list_peek = w[:5, :5, :5, :5].tolist()
    #             else:
    #                 list_peek = []
    #             print(f"{t.name:<60}: {w.shape} "
    #                   f"{list_peek}")

    # print(model.summary())
    # print(model.layers[1].weights[0].numpy().shape)
    # print(model.layers[1].bias)
    # print(model.get)


def main():
    # model = tf.keras.applications.mobilenet.MobileNet(
    #     input_shape=None,
    #     alpha=1.0,
    #     depth_multiplier=1,
    #     dropout=0.001,
    #     include_top=True,
    #     weights='imagenet',
    #     input_tensor=None,
    #     pooling=None,
    #     classes=1000,
    #     classifier_activation='softmax',
    # )
    # build_pytorch_network_state(model)

    # peek_tensorflow_state()
    peek_pytorch_state()

    # converter = ConverterTensorFlow()
    # converter.build_tf_model()
    # converter.convert_state()
    # converter.save_to(r".\pretrained")

    # converter = ConverterTensorFlow(alpha=0.5, input_resolution=160)
    # converter.build_tf_model()
    # converter.convert_state()
    # converter.save_to(r".\pretrained")

    # model_path = r".\pretrained\wjc852456\mobilenet_sgd_rmsprop_69.526.tar"
    # out_dir = r".\pretrained"
    # name = "wjc2_rmsprop"
    #
    # converter = ConverterWJC852456()
    #
    # print(f"loading wjc model in {model_path} ...")
    # converter.load_wjc852456_model(model_path)
    #
    # print(f"converting to pytorch state dict ...")
    # converter.convert_state()
    #
    # converter.save_to(out_dir, name=name)
    # print(f"model file saved to {converter.filename}")


if __name__ == '__main__':
    main()
