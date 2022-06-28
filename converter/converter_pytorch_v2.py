from pathlib import Path

import torch
from torchvision.models import mobilenet_v2

from script.utils import defaultdict_none


class ConverterPyTorchV2:
    def __init__(self):
        self.repeats = [1, 2, 3, 4, 3, 3, 1]
        self.alpha = 1.0
        self.input_resolution = 224
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
        self._pt2my_conv(f"features.1.conv.0.0", f"bottlenecks.0.block.0.dw_conv2.conv")
        self._pt2my_bn(f"features.1.conv.0.1", f"bottlenecks.0.block.0.bn2")
        self._pt2my_conv(f"features.1.conv.1", f"bottlenecks.0.block.0.pw_conv3.conv")
        self._pt2my_bn(f"features.1.conv.2", f"bottlenecks.0.block.0.bn3")
        pt_idx = 2
        for my_idx1 in range(1, len(self.repeats)):
            for my_idx2 in range(self.repeats[my_idx1]):
                self._pt2my_conv(f"features.{pt_idx}.conv.0.0",
                                 f"bottlenecks.{my_idx1}.block.{my_idx2}.pw_conv1.conv")
                self._pt2my_bn(f"features.{pt_idx}.conv.0.1",
                               f"bottlenecks.{my_idx1}.block.{my_idx2}.bn1")
                self._pt2my_conv(f"features.{pt_idx}.conv.1.0",
                                 f"bottlenecks.{my_idx1}.block.{my_idx2}.dw_conv2.conv")
                self._pt2my_bn(f"features.{pt_idx}.conv.1.1",
                               f"bottlenecks.{my_idx1}.block.{my_idx2}.bn2")
                self._pt2my_conv(f"features.{pt_idx}.conv.2",
                                 f"bottlenecks.{my_idx1}.block.{my_idx2}.pw_conv3.conv")
                self._pt2my_bn(f"features.{pt_idx}.conv.3",
                               f"bottlenecks.{my_idx1}.block.{my_idx2}.bn3")
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
        self.filename = out_path / f"{name}-mobilenetv2-a{self.alpha * 100:.0f}-r{self.input_resolution:.0f}" \
                                   f"-c{self.num_class}-e{0:04d}.pth"
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
