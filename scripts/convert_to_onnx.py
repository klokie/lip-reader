"""
Convert LipCoordNet PyTorch weights to ONNX format and quantize for browser delivery.

Usage:
    python scripts/convert_to_onnx.py

Outputs:
    public/models/lipcoordnet.onnx          (fp32)
    public/models/lipcoordnet_int8.onnx     (quantized)
"""

import os
import sys
import urllib.request
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

WEIGHTS_URL = (
    "https://huggingface.co/SilentSpeak/LipCoordNet/resolve/main/"
    "pretrain/LipCoordNet_coords_loss_0.025581153109669685"
    "_wer_0.01746208431890914_cer_0.006488426950253695.pt"
)
WEIGHTS_PATH = "scripts/lipcoordnet_weights.pt"
OUTPUT_DIR = "public/models"
ONNX_PATH = os.path.join(OUTPUT_DIR, "lipcoordnet.onnx")
ONNX_INT8_PATH = os.path.join(OUTPUT_DIR, "lipcoordnet_int8.onnx")


class LipCoordNet(nn.Module):
    """Exact replica of the upstream model architecture."""

    def __init__(self, dropout_p=0.5, coord_input_dim=40, coord_hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512 + 2 * coord_hidden_dim, 27 + 1)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)

        self.coord_gru = nn.GRU(
            coord_input_dim, coord_hidden_dim, 1, bidirectional=True
        )

        self._init()

    def _init(self):
        init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        init.constant_(self.conv1.bias, 0)
        init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        init.constant_(self.conv2.bias, 0)
        init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        init.constant_(self.conv3.bias, 0)
        init.kaiming_normal_(self.FC.weight, nonlinearity="sigmoid")
        init.constant_(self.FC.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(
                    m.weight_ih_l0[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0[i : i + 256])
                init.constant_(m.bias_ih_l0[i : i + 256], 0)
                init.uniform_(
                    m.weight_ih_l0_reverse[i : i + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0_reverse[i : i + 256])
                init.constant_(m.bias_ih_l0_reverse[i : i + 256], 0)

    def forward(self, x, coords):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)

        # (B, C, T, H, W) -> (T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x)

        self.coord_gru.flatten_parameters()

        # (B, T, N, 2) -> (T, B, N*2)
        coords = coords.permute(1, 0, 2, 3).contiguous()
        coords = coords.view(coords.size(0), coords.size(1), -1)
        coords, _ = self.coord_gru(coords)
        coords = self.dropout(coords)

        combined = torch.cat((x, coords), dim=2)
        x = self.FC(combined)
        # (T, B, C) -> (B, T, C)
        x = x.permute(1, 0, 2).contiguous()
        return x


def download_weights():
    if os.path.exists(WEIGHTS_PATH):
        print(f"Weights already at {WEIGHTS_PATH}")
        return
    print(f"Downloading weights from HuggingFace...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
    print(f"Downloaded: {size_mb:.1f} MB")


def convert_to_onnx():
    print("Loading PyTorch model...")
    model = LipCoordNet()
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Video: [batch, channels(3), frames(75), height(64), width(128)]
    # Coords: [batch, frames(75), landmarks(20), xy(2)]
    dummy_video = torch.randn(1, 3, 75, 64, 128)
    dummy_coords = torch.randn(1, 75, 20, 2)

    print("Verifying forward pass...")
    with torch.no_grad():
        out = model(dummy_video, dummy_coords)
    print(f"Output shape: {out.shape} (expected [1, 75, 28])")
    assert out.shape == (1, 75, 28), f"Unexpected shape: {out.shape}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Exporting to ONNX: {ONNX_PATH}")
    torch.onnx.export(
        model,
        (dummy_video, dummy_coords),
        ONNX_PATH,
        input_names=["video", "coords"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"ONNX model valid: {size_mb:.1f} MB")
    return size_mb


def quantize():
    print(f"Quantizing to int8: {ONNX_INT8_PATH}")
    quantize_dynamic(
        ONNX_PATH,
        ONNX_INT8_PATH,
        weight_type=QuantType.QUInt8,
    )
    size_mb = os.path.getsize(ONNX_INT8_PATH) / (1024 * 1024)
    print(f"Quantized model: {size_mb:.1f} MB")
    return size_mb


def validate_quantized():
    import onnxruntime as ort

    print("Validating quantized model with ONNX Runtime...")
    sess = ort.InferenceSession(ONNX_INT8_PATH)

    video = np.random.randn(1, 3, 75, 64, 128).astype(np.float32)
    coords = np.random.randn(1, 75, 20, 2).astype(np.float32)

    outputs = sess.run(None, {"video": video, "coords": coords})
    print(f"Quantized output shape: {outputs[0].shape} (expected (1, 75, 28))")
    assert outputs[0].shape == (1, 75, 28), f"Unexpected: {outputs[0].shape}"
    print("Validation passed.")


if __name__ == "__main__":
    download_weights()
    fp32_size = convert_to_onnx()
    int8_size = quantize()
    validate_quantized()

    print(f"\n--- Summary ---")
    print(f"fp32 ONNX:  {fp32_size:.1f} MB  ({ONNX_PATH})")
    print(f"int8 ONNX:  {int8_size:.1f} MB  ({ONNX_INT8_PATH})")
    print(f"Reduction:  {(1 - int8_size / fp32_size) * 100:.0f}%")
