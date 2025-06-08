import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import Upsample  # ⬅️ Fix: Upsample 추가
from .common import Conv
from models.common import (
    C3, SPPF, Concat, Bottleneck, C3Ghost, C3TR, C3x,
    GhostConv, GhostBottleneck, DWConv, SPP, MobileNetV2_Backbone
)
from models.experimental import MixConv2d
from models.mobilenetv2 import MobileNetV2_Backbone
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_yaml, make_divisible
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, time_sync

try:
    import thop
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None
    dynamic = False
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        print("anchors:", type(anchors), anchors)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
        return x if self.training else (torch.cat(z, 1),)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand_as(grid)
        return grid, anchor_grid


class BaseModel(nn.Module):
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        super().__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)

        # 클래스 수 덮어쓰기
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc

        # anchors 덮어쓰기
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = anchors

        # anchors가 문자열이면 파싱
        import ast
        if isinstance(self.yaml["anchors"], str):
            self.yaml["anchors"] = ast.literal_eval(self.yaml["anchors"])

        # 모델 생성
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml["nc"])]
        self.inplace = self.yaml.get("inplace", True)

        # Detect 모듈 초기화
        m = self.model[-1]
        if isinstance(m, Detect):
            def _forward(x): return self.forward(x)[0]
            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()

        initialize_weights(self)
        self.info()

    def forward(self, x, augment=False, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)

    def _initialize_biases(self, cf=None):
        m = self.model[-1]  # Detect layer
        for mi, s in zip(m.m, m.stride):  # per output layer
            b = mi.bias.view(m.na, -1)
            # obj bias
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            # cls bias
            b.data[:, 5 : 5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)



Model = DetectionModel


def parse_model(d, ch):
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"]
    layers, save, c2 = [], [], ch[-1]

    layer_map = {
        'Conv': Conv,
        'C3': C3,
        'SPPF': SPPF,
        'Concat': Concat,
        'MobileNetV2_Backbone': MobileNetV2_Backbone,
        'Detect': Detect,
        'Upsample': Upsample,
        'nn.BatchNorm2d': nn.BatchNorm2d,
        'Bottleneck': Bottleneck,
        'DWConv': DWConv,
        'GhostConv': GhostConv,
        'GhostBottleneck': GhostBottleneck,
        'MixConv2d': MixConv2d,
        'C3Ghost': C3Ghost,
        'C3TR': C3TR,
        'C3x': C3x,
        'SPP': SPP,
    }

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        if isinstance(m, str):
            if m in layer_map:
                m = layer_map[m]
            else:
                raise ValueError(f"Layer '{m}' not found in custom layer_map.")

        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a, layer_map) if isinstance(a, str) else a

        n = max(round(n * gd), 1) if n > 1 else n
        if m in {Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, C3, C3Ghost, C3TR, C3x}:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in {C3, C3Ghost, C3TR, C3x}:
                args.insert(2, n)
                n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info(f"{i:>3}{str(f):>18}{n:>3}{np:10.0f}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

if __name__ == "__main__":
    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ]
    model = DetectionModel(cfg='models/mobile-yolo5s_voc.yaml', ch=3, nc=31, anchors=anchors)
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)

class DetectMultiBackend(nn.Module):
    def __init__(self, model):
        super(DetectMultiBackend, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)