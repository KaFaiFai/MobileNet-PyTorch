from pathlib import Path

import torch

from script.utils import defaultdict_none


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
