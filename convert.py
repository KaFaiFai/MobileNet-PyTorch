"""
convert from
- MobileNetTF: official tensorflow pretrained model
- MobileNet: wjc852456's repo https://github.com/wjc852456/pytorch-mobilenet-v1
- MobileNetV2: official pytorch pretrained model
"""
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2
import tensorflow as tf
import numpy as np

from model import *
from script.utils import return_none, defaultdict_none


class ConverterTensorFlow:
    def __init__(self, alpha=1.0, input_resolution=224, num_class=1000):
        self.alpha = alpha
        self.input_resolution = input_resolution
        self.num_class = num_class
        self.filename = None
        self._model = None
        self._state_dict = dict()

    def build_tf_model(self):
        if self.input_resolution != 224:
            raise Exception("Not implemented for input_resolution != 224")

        self._model = tf.keras.applications.mobilenet.MobileNet(
            input_shape=(self.input_resolution, self.input_resolution, 3),
            alpha=self.alpha,
            depth_multiplier=1,
            dropout=0.001,
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            pooling=None,
            classes=self.num_class,
            classifier_activation='softmax',
        )

    def convert_state(self):
        assert self._model is not None, "model is not loaded yet. Please call build_tf_model() first"

        self._state_dict = dict()

        # initial
        self._tf2my_conv("conv1", "initial.1")
        self._tf2my_bn("conv1_bn", "initial.2")

        # separable_convs
        for tf_idx, my_idx in zip(range(1, 14), range(13)):
            self._tf2my_conv_dw(f"conv_dw_{tf_idx}", f"separable_convs.{my_idx}.dw_conv.conv")
            self._tf2my_bn(f"conv_dw_{tf_idx}_bn", f"separable_convs.{my_idx}.bn1")
            self._tf2my_conv(f"conv_pw_{tf_idx}", f"separable_convs.{my_idx}.pw_conv.conv")
            self._tf2my_bn(f"conv_pw_{tf_idx}_bn", f"separable_convs.{my_idx}.bn2")

        # final
        self._tf2my_conv("conv_preds", "final.2", has_bias=True)

        # for param_tensor in self._state_dict:
        #     print(f"{param_tensor:<60}: {self._state_dict[param_tensor].size()}")

    def save_to(self, out_dir, name="tf"):
        assert self._state_dict is not None, "state is not loaded yet. Please call convert_state() first"

        state = defaultdict_none({"epoch": -1, "alpha": self.alpha, "input_resolution": self.input_resolution,
                                  "num_class": self.num_class, "state_dict": self._state_dict})
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True, parents=True)
        self.filename = out_path / f"{name}-mobilenet-a{self.alpha * 100:.0f}-r{self.input_resolution:.0f}" \
                                   f"-c{self.num_class}-e{0:04d}.pth"
        torch.save(state, str(self.filename))

    def _tf2my_conv(self, tf_layer: str, my_layer: str, has_bias: bool = False):
        # conv2d kernel shape: pytorch(OUT_C, IN_C, H, W), tensorflow(H, W, IN_C, OUT_C)
        tf_weights = self._model.get_layer(tf_layer).get_weights()
        self._state_dict[f"{my_layer}.weight"] = torch.from_numpy(np.transpose(tf_weights[0], (3, 2, 0, 1)))
        if has_bias:
            self._state_dict[f"{my_layer}.bias"] = torch.from_numpy(tf_weights[1])

    def _tf2my_conv_dw(self, tf_layer: str, my_layer: str):
        # conv2d kernel shape: pytorch(OUT_C, IN_C, H, W), tensorflow(H, W, OUT_C, IN_C)
        tf_kernel = self._model.get_layer(tf_layer).get_weights()[0]
        self._state_dict[f"{my_layer}.weight"] = torch.from_numpy(np.transpose(tf_kernel, (2, 3, 0, 1)))

    def _tf2my_bn(self, tf_layer: str, my_layer: str):
        # tf weights: [gamma, beta, moving_mean, moving_variance]
        # my params : [.weight, .bias, .running_mean, .running_var, .num_batches_tracked]
        tf_weights = self._model.get_layer(tf_layer).get_weights()
        self._state_dict[f"{my_layer}.weight"] = torch.from_numpy(tf_weights[0])
        self._state_dict[f"{my_layer}.bias"] = torch.from_numpy(tf_weights[1])
        self._state_dict[f"{my_layer}.running_mean"] = torch.from_numpy(tf_weights[2])
        self._state_dict[f"{my_layer}.running_var"] = torch.from_numpy(tf_weights[3])
        self._state_dict[f"{my_layer}.num_batches_tracked"] = torch.tensor(0)


class ConverterWJC852456:
    def __init__(self):
        self.alpha = 1.0
        self.input_resolution = 224
        self.num_class = 1000
        self.filename = None
        self._model = None
        self._state_dict = None

    def load_wjc852456_model(self, model_path):
        if self.input_resolution != 224:
            raise Exception("Not implemented for input_resolution != 224")

        state = torch.load(model_path)
        self._model = state["state_dict"]

    def convert_state(self):
        assert self._model is not None, "model is not loaded yet. Please call load_wjc852456_model() first"

        self._state_dict = dict()

        # initial
        self._wjc2my_conv("module.model.0.0", "initial.1")
        self._wjc2my_bn("module.model.0.1", "initial.2")

        # separable_convs
        for wjc_idx, my_idx in zip(range(1, 14), range(13)):
            self._wjc2my_conv(f"module.model.{wjc_idx}.0", f"separable_convs.{my_idx}.dw_conv.conv")
            self._wjc2my_bn(f"module.model.{wjc_idx}.1", f"separable_convs.{my_idx}.bn1")
            self._wjc2my_conv(f"module.model.{wjc_idx}.3", f"separable_convs.{my_idx}.pw_conv.conv")
            self._wjc2my_bn(f"module.model.{wjc_idx}.4", f"separable_convs.{my_idx}.bn2")

        # final
        self._wjc2my_fc("module.fc", "final.2")

        # for param_tensor in self._state_dict:
        #     print(f"{param_tensor:<60}: {self._state_dict[param_tensor].size()}")

    def save_to(self, out_dir, name="wjc"):
        assert self._state_dict is not None, "state is not loaded yet. Please call convert_state() first"

        state = defaultdict_none({"epoch": -1, "alpha": self.alpha, "input_resolution": self.input_resolution,
                                  "num_class": self.num_class, "state_dict": self._state_dict})
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True, parents=True)
        self.filename = out_path / f"{name}-mobilenet-a{self.alpha * 100:.0f}-r{self.input_resolution:.0f}" \
                                   f"-c{self.num_class}-e{0:04d}.pth"
        torch.save(state, str(self.filename))

    def _wjc2my_conv(self, wjc_layer: str, my_layer: str):
        self._state_dict[f"{my_layer}.weight"] = self._model[f"{wjc_layer}.weight"]

    def _wjc2my_bn(self, wjc_layer: str, my_layer: str):
        self._state_dict[f"{my_layer}.weight"] = self._model[f"{wjc_layer}.weight"]
        self._state_dict[f"{my_layer}.bias"] = self._model[f"{wjc_layer}.bias"]
        self._state_dict[f"{my_layer}.running_mean"] = self._model[f"{wjc_layer}.running_mean"]
        self._state_dict[f"{my_layer}.running_var"] = self._model[f"{wjc_layer}.running_var"]
        self._state_dict[f"{my_layer}.num_batches_tracked"] = torch.tensor(0)

    def _wjc2my_fc(self, wjc_layer: str, my_layer: str):
        wjc_weight = self._model[f"{wjc_layer}.weight"]
        self._state_dict[f"{my_layer}.weight"] = wjc_weight.view(*wjc_weight.shape, 1, 1)
        self._state_dict[f"{my_layer}.bias"] = self._model[f"{wjc_layer}.bias"]


class ConverterPyTorchV2:
    def __init__(self):
        self.repeats = [1, 2, 3, 4, 3, 3, 1]
        self.num_class = 1000
        self.filename = None
        self._model = None
        self._state_dict = None

    def build_pt_model(self):
        self._model = mobilenet_v2(pretrained=True).state_dict()

    def convert_state(self):
        assert self._model is not None, "model is not loaded yet. Please call load_wjc852456_model() first"

        self._state_dict = dict()

        # initial
        self._pt2my_conv("features.0.0", "initial.1")
        self._pt2my_bn("features.0.1", "initial.2")

        # bottlenecks, first layer is different
        self._pt2my_conv(f"features.1.conv.0.0", f"separable_convs.0.block.0.dw_conv2.conv")
        self._pt2my_bn(f"features.1.conv.0.1", f"separable_convs.0.block.0.bn2")
        self._pt2my_conv(f"features.1.conv.1", f"separable_convs.0.block.0.pw_conv3.conv")
        self._pt2my_bn(f"features.1.conv.2", f"separable_convs.0.block.0.bn3")
        pt_idx = 2
        for my_idx1 in range(1, len(self.repeats)):
            for my_idx2 in range(self.repeats[my_idx1]):
                self._pt2my_conv(f"features.{pt_idx}.conv.0.0",
                                 f"separable_convs.{my_idx1}.block.{my_idx2}.pw_conv1.conv")
                self._pt2my_bn(f"features.{pt_idx}.conv.0.1",
                               f"separable_convs.{my_idx1}.block.{my_idx2}.bn1")
                self._pt2my_conv(f"features.{pt_idx}.conv.1.0",
                                 f"separable_convs.{my_idx1}.block.{my_idx2}.dw_conv2.conv")
                self._pt2my_bn(f"features.{pt_idx}.conv.1.1",
                               f"separable_convs.{my_idx1}.block.{my_idx2}.bn2")
                self._pt2my_conv(f"features.{pt_idx}.conv.2",
                                 f"separable_convs.{my_idx1}.block.{my_idx2}.pw_conv3.conv")
                self._pt2my_bn(f"features.{pt_idx}.conv.3",
                               f"separable_convs.{my_idx1}.block.{my_idx2}.bn3")
                pt_idx += 1

        # final
        self._pt2my_conv("features.18.0", "final.0.conv")
        self._pt2my_bn("features.18.1", "final.1")
        self._pt2my_fc("classifier.1", "final.6")

        # for param_tensor in self._state_dict:
        #     print(f"{param_tensor:<60}: {self._state_dict[param_tensor].size()}")

    def save_to(self, out_dir, name="pt"):
        assert self._state_dict is not None, "state is not loaded yet. Please call convert_state() first"
        state = defaultdict_none({"epoch": -1, "num_class": self.num_class, "state_dict": self._state_dict})
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True, parents=True)
        self.filename = out_path / f"{name}-mobilenetv2-c{self.num_class}-e{0:04d}.pth"
        torch.save(state, str(self.filename))

    def _pt2my_conv(self, pt_layer: str, my_layer: str):
        self._state_dict[f"{my_layer}.weight"] = self._model[f"{pt_layer}.weight"]

    def _pt2my_bn(self, pt_layer: str, my_layer: str):
        self._state_dict[f"{my_layer}.weight"] = self._model[f"{pt_layer}.weight"]
        self._state_dict[f"{my_layer}.bias"] = self._model[f"{pt_layer}.bias"]
        self._state_dict[f"{my_layer}.running_mean"] = self._model[f"{pt_layer}.running_mean"]
        self._state_dict[f"{my_layer}.running_var"] = self._model[f"{pt_layer}.running_var"]
        self._state_dict[f"{my_layer}.num_batches_tracked"] = self._model[f"{pt_layer}.num_batches_tracked"]

    def _pt2my_fc(self, pt_layer: str, my_layer: str):
        self._state_dict[f"{my_layer}.weight"] = self._model[f"{pt_layer}.weight"]
        self._state_dict[f"{my_layer}.bias"] = self._model[f"{pt_layer}.bias"]


def peek_pytorch_state():
    network = MobileNetV2(1000)
    network_state = network.state_dict()
    network = mobilenet_v2(pretrained=True)
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
    # peek_pytorch_state()

    # converter = ConverterTensorFlow()
    # converter.build_tf_model()
    # converter.convert_state()
    # converter.save_to(r".\pretrained")

    converter = ConverterPyTorchV2()
    converter.build_pt_model()
    converter.convert_state()
    converter.save_to(r".\pretrained")

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
