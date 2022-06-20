"""
convert from
- official tensorflow pretrained model (WIP)
- wjc852456's repo https://github.com/wjc852456/pytorch-mobilenet-v1
"""
from pathlib import Path

import torch
import tensorflow as tf
import numpy as np

from model import MobileNet


class ConverterOfficial:
    def __init__(self, alpha=1.0, input_resolution=224, num_class=1000):
        raise Exception("Not implemented")
        self.alpha = alpha
        self.input_resolution = input_resolution
        self.num_class = num_class
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

    def to_pytorch_state(self):
        self._state_dict = dict()

        # initial
        self._tf2pt_conv("conv1", "initial.0")
        self._tf2pt_bn("conv1_bn", "initial.1")

        # separable_convs
        for tf_idx, pt_idx in zip(range(1, 14), range(13)):
            self._tf2pt_conv_dw(f"conv_dw_{tf_idx}", f"separable_convs.{pt_idx}.dw_conv.conv")
            self._tf2pt_bn(f"conv_dw_{tf_idx}_bn", f"separable_convs.{pt_idx}.bn1")
            self._tf2pt_conv(f"conv_pw_{tf_idx}", f"separable_convs.{pt_idx}.pw_conv.conv")
            self._tf2pt_bn(f"conv_pw_{tf_idx}_bn", f"separable_convs.{pt_idx}.bn2")

        # final
        self._tf2pt_conv("conv_preds", "final.2", has_bias=True)

        # for param_tensor in self._state_dict:
        #     print(f"{param_tensor:<45}: {self._state_dict[param_tensor].size()}")

    def save_to(self, out_dir):
        state = {"epoch": -1, "alpha": self.alpha, "input_resolution": self.input_resolution,
                 "num_class": self.num_class, "state_dict": self._state_dict}
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True, parents=True)
        save_to = out_path / f"mobile_net-a{self.alpha * 100:.0f}-r{self.input_resolution:.0f}" \
                             f"-c{self.num_class}-e{0:04d}.pth"
        torch.save(state, str(save_to))

    def _tf2pt_conv(self, tf_layer: str, pt_layer: str, has_bias: bool = False):
        # conv2d kernel shape: pytorch(OUT_C, IN_C, H, W), tensorflow(H, W, IN_C, OUT_C)
        tf_weights = self._model.get_layer(tf_layer).get_weights()
        self._state_dict[f"{pt_layer}.weight"] = torch.from_numpy(np.transpose(tf_weights[0], (3, 2, 0, 1)))
        if has_bias:
            self._state_dict[f"{pt_layer}.bias"] = torch.from_numpy(tf_weights[1])

    def _tf2pt_conv_dw(self, tf_layer: str, pt_layer: str):
        # conv2d kernel shape: pytorch(OUT_C, IN_C, H, W), tensorflow(H, W, OUT_C, IN_C)
        tf_kernel = self._model.get_layer(tf_layer).get_weights()[0]
        self._state_dict[f"{pt_layer}.weight"] = torch.from_numpy(np.transpose(tf_kernel, (2, 3, 0, 1)))

    def _tf2pt_bn(self, tf_layer: str, pt_layer: str):
        # tf weights: [gamma, beta, moving_mean, moving_variance]
        # pt params : [.weight, .bias, .running_mean, .running_var, .num_batches_tracked]
        tf_weights = self._model.get_layer(tf_layer).get_weights()
        self._state_dict[f"{pt_layer}.weight"] = torch.from_numpy(tf_weights[0])
        self._state_dict[f"{pt_layer}.bias"] = torch.from_numpy(tf_weights[1])
        self._state_dict[f"{pt_layer}.running_mean"] = torch.from_numpy(tf_weights[2])
        self._state_dict[f"{pt_layer}.running_var"] = torch.from_numpy(tf_weights[3])
        self._state_dict[f"{pt_layer}.num_batches_tracked"] = torch.tensor(0)


class ConverterWJC852456:
    def __init__(self):
        self.alpha = 1.0
        self.input_resolution = 224
        self.num_class = 1000
        self._model = None
        self._state_dict = dict()

    def load_wjc852456_model(self):
        if self.input_resolution != 224:
            raise Exception("Not implemented for input_resolution != 224")

        state = torch.load(r"C:\_Project\Pycharm Projects\MobileNet\pretrained\mobilenet_sgd_68.848.pth.tar")
        self._model = state["state_dict"]

    def to_pytorch_state(self):
        self._state_dict = dict()

        # initial
        self._wjc2pt_conv("module.model.0.0", "initial.0")
        self._wjc2pt_bn("module.model.0.1", "initial.1")

        # separable_convs
        for wjc_idx, pt_idx in zip(range(1, 14), range(13)):
            self._wjc2pt_conv(f"module.model.{wjc_idx}.0", f"separable_convs.{pt_idx}.dw_conv.conv")
            self._wjc2pt_bn(f"module.model.{wjc_idx}.1", f"separable_convs.{pt_idx}.bn1")
            self._wjc2pt_conv(f"module.model.{wjc_idx}.3", f"separable_convs.{pt_idx}.pw_conv.conv")
            self._wjc2pt_bn(f"module.model.{wjc_idx}.4", f"separable_convs.{pt_idx}.bn2")

        # final
        self._wjc2pt_fc("module.fc", "final.2")

        for param_tensor in self._state_dict:
            print(f"{param_tensor:<45}: {self._state_dict[param_tensor].size()}")

    def save_to(self, out_dir):
        state = {"epoch": -1, "alpha": self.alpha, "input_resolution": self.input_resolution,
                 "num_class": self.num_class, "state_dict": self._state_dict}
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True, parents=True)
        save_to = out_path / f"wjc-mobilenet-a{self.alpha * 100:.0f}-r{self.input_resolution:.0f}" \
                             f"-c{self.num_class}-e{0:04d}.pth"
        torch.save(state, str(save_to))

    def _wjc2pt_conv(self, wjc_layer: str, pt_layer: str):
        self._state_dict[f"{pt_layer}.weight"] = self._model[f"{wjc_layer}.weight"]

    def _wjc2pt_bn(self, wjc_layer: str, pt_layer: str):
        self._state_dict[f"{pt_layer}.weight"] = self._model[f"{wjc_layer}.weight"]
        self._state_dict[f"{pt_layer}.bias"] = self._model[f"{wjc_layer}.bias"]
        self._state_dict[f"{pt_layer}.running_mean"] = self._model[f"{wjc_layer}.running_mean"]
        self._state_dict[f"{pt_layer}.running_var"] = self._model[f"{wjc_layer}.running_var"]
        self._state_dict[f"{pt_layer}.num_batches_tracked"] = torch.tensor(0)

    def _wjc2pt_fc(self, wjc_layer: str, pt_layer: str):
        wjc_weight = self._model[f"{wjc_layer}.weight"]
        self._state_dict[f"{pt_layer}.weight"] = wjc_weight.view(*wjc_weight.shape, 1, 1)
        self._state_dict[f"{pt_layer}.bias"] = self._model[f"{wjc_layer}.bias"]


def peek_pytorch_state():
    network = MobileNet(1000)
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
        print(f"{param_tensor:<45}: {network_state[param_tensor].size()}")
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
    #             print(f"{t.name:<45}: {w.shape} "
    #                   f"{list_peek}")

    # print(model.summary())
    # print(model.layers[1].weights[0].numpy().shape)
    # print(model.layers[1].bias)
    # print(model.get)


if __name__ == '__main__':
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

    # converter = ConverterOfficial()
    # converter.build_tf_model()
    # converter.to_pytorch_state()
    # converter.save_to(r".\pretrained")

    converter = ConverterWJC852456()
    converter.load_wjc852456_model()
    converter.to_pytorch_state()
    converter.save_to(r".\pretrained")
