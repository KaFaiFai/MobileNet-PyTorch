from pathlib import Path

import numpy as np
import torch
import tensorflow as tf

from script.utils import defaultdict_none


class ConverterTensorFlow:
    def __init__(self, alpha=1.0, input_resolution=224):
        self.alpha = alpha
        self.input_resolution = input_resolution
        self.num_class = 1000
        self.filename = None
        self._model = None
        self._state_dict = None

    def build_tf_model(self):
        """
        (input_resolution, alpha) in [128, 160, 192, 224] x [0.25, 0.50, 0.75, 1.0]
        """
        if self.alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise Exception(f"alpha can only be one of 0.25, 0.50, 0.75 or 1.0, but got alpha={self.alpha}")
        if self.input_resolution not in [128, 160, 192, 224]:
            raise Exception(f"input_resolution can only be one of 128, 160, 192 or 224, "
                            f"but got input_resolution={self.input_resolution}")

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
