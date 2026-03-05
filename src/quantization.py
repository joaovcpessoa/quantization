from __future__ import annotations

import io
import math
import time
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchvision import datasets, transforms
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. TIPOS E CONSTANTES
# ─────────────────────────────────────────────

BitWidth = Literal[1, 2, 4, 8]
DatasetName = Literal["cifar10", "cifar100", "tiny_imagenet", "mnist"]

def _int_range(bits: int, signed: bool = True) -> Tuple[int, int]:
    """
    Retorna (q_min, q_max) para um inteiro de `bits` bits.

    Simétrico/signed:   [-2^(b-1)+1,  2^(b-1)-1]   (exclui -128 para ser simétrico)
    Assimétrico/unsigned: [0, 2^b - 1]
    """
    if signed:
        q_max = 2 ** (bits - 1) - 1      # ex: INT8 → 127
        q_min = -(2 ** (bits - 1)) + 1   # ex: INT8 → -127  (simétrico)
    else:
        q_min = 0
        q_max = 2 ** bits - 1            # ex: INT8 unsigned → 255
    return q_min, q_max


# ═══════════════════════════════════════════════════════════════
# 2. MÓDULO DE QUANTIZAÇÃO  (núcleo matemático)
# ═══════════════════════════════════════════════════════════════

class QuantizationMath:
    """
    Cálculos de baixo nível totalmente manuais.
    Sem qualquer dependência de APIs de quantização.
    """

    # ─── 2.1 Scale e Zero-Point ───────────────────────────────

    @staticmethod
    def compute_scale_zp_asymmetric(
        x_min: float,
        x_max: float,
        bits: int,
    ) -> Tuple[float, int]:
        """
        Quantização ASSIMÉTRICA (affine).

        S = (x_max - x_min) / (q_max - q_min)
        Z = round(q_min - x_min / S)

        Garante que zero real → Z no espaço inteiro.
        """
        q_min, q_max = _int_range(bits, signed=False)   # usa unsigned para assimétrico
        # evita divisão por zero
        x_range = float(x_max) - float(x_min)
        if x_range < 1e-8:
            x_range = 1e-8

        scale = x_range / (q_max - q_min)
        zero_point = int(round(q_min - float(x_min) / scale))
        zero_point = int(max(q_min, min(q_max, zero_point)))   # clamp
        return scale, zero_point

    @staticmethod
    def compute_scale_symmetric(
        x_abs_max: float,
        bits: int,
    ) -> Tuple[float, int]:
        """
        Quantização SIMÉTRICA (Z = 0).

        S = x_abs_max / q_max

        Mais eficiente: multiplicação por Z some no forward.
        """
        q_min, q_max = _int_range(bits, signed=True)
        if x_abs_max < 1e-8:
            x_abs_max = 1e-8
        scale = float(x_abs_max) / q_max
        return scale, 0   # zero-point sempre 0

    # ─── 2.2 Quantizar / Dequantizar ──────────────────────────

    @staticmethod
    def quantize(
        x: Tensor,
        scale: float,
        zero_point: int,
        bits: int,
        signed: bool,
    ) -> Tensor:
        """
        Aplica quantização:
          Q(x) = clamp( round(x/S) + Z,  q_min, q_max )
        """
        q_min, q_max = _int_range(bits, signed=signed)
        x_scaled = x / scale + zero_point      # x/S + Z
        x_rounded = torch.round(x_scaled)      # arredondamento para inteiro mais próximo
        x_clamped = torch.clamp(x_rounded, q_min, q_max)   # clipping
        return x_clamped

    @staticmethod
    def dequantize(
        x_q: Tensor,
        scale: float,
        zero_point: int,
    ) -> Tensor:
        """
        Reconstrói o float:
          x̂ = S * (Q - Z)
        """
        return scale * (x_q - zero_point)

    @staticmethod
    def fake_quantize(
        x: Tensor,
        scale: float,
        zero_point: int,
        bits: int,
        signed: bool,
    ) -> Tensor:
        """
        Fake-quantization (quantiza e dequantiza no mesmo passo).
        Mantém o tensor em float mas simula os erros de quantização.
        Usado no QAT.

          x̂ = S * (clamp(round(x/S + Z), q_min, q_max) - Z)
        """
        x_q = QuantizationMath.quantize(x, scale, zero_point, bits, signed)
        return QuantizationMath.dequantize(x_q, scale, zero_point)


# ═══════════════════════════════════════════════════════════════
# 3. OBSERVADORES  (coletam estatísticas dos ativações/pesos)
# ═══════════════════════════════════════════════════════════════

class MinMaxObserver:
    """
    Observa o mínimo e máximo de tensores ao longo do tempo.
    Usado durante a calibração do PTQ ou ao longo do forward no QAT.
    """

    def __init__(self):
        self.min_val: Optional[Tensor] = None
        self.max_val: Optional[Tensor] = None

    def update(self, x: Tensor):
        x_min = x.detach().min()
        x_max = x.detach().max()
        if self.min_val is None:
            self.min_val = x_min
            self.max_val = x_max
        else:
            self.min_val = torch.min(self.min_val, x_min)
            self.max_val = torch.max(self.max_val, x_max)

    def reset(self):
        self.min_val = None
        self.max_val = None


class PerChannelObserver:
    """
    Observa min/max por canal de saída.
    Para pesos de camadas Linear: canal = linha da matriz de peso.
    """

    def __init__(self):
        self.min_vals: Optional[Tensor] = None
        self.max_vals: Optional[Tensor] = None

    def update(self, x: Tensor):
        """x shape: [out_features, in_features]"""
        x_min = x.detach().min(dim=1).values    # mínimo por linha (canal de saída)
        x_max = x.detach().max(dim=1).values
        if self.min_vals is None:
            self.min_vals = x_min
            self.max_vals = x_max
        else:
            self.min_vals = torch.min(self.min_vals, x_min)
            self.max_vals = torch.max(self.max_vals, x_max)

    def reset(self):
        self.min_vals = None
        self.max_vals = None


# ═══════════════════════════════════════════════════════════════
# 4. QUANTIZADORES DE ALTO NÍVEL
# ═══════════════════════════════════════════════════════════════

class TensorQuantizer:
    """
    Quantizador per-tensor: um único par (scale, zero_point) para todo o tensor.
    Suporta simétrico e assimétrico.
    """

    def __init__(
        self,
        bits: int,
        symmetric: bool = True,
    ):
        self.bits = bits
        self.symmetric = symmetric
        self.scale: Optional[float] = None
        self.zero_point: Optional[int] = None

    def calibrate(self, x: Tensor):
        """Calcula scale e zero-point a partir do tensor."""
        if self.symmetric:
            abs_max = x.detach().abs().max().item()
            self.scale, self.zero_point = QuantizationMath.compute_scale_symmetric(
                abs_max, self.bits
            )
        else:
            x_min = x.detach().min().item()
            x_max = x.detach().max().item()
            self.scale, self.zero_point = QuantizationMath.compute_scale_zp_asymmetric(
                x_min, x_max, self.bits
            )

    def quantize_dequantize(self, x: Tensor) -> Tensor:
        """Fake-quantization: simula o erro sem sair do domínio float."""
        assert self.scale is not None, "Chame .calibrate() antes de quantizar."
        signed = self.symmetric
        return QuantizationMath.fake_quantize(
            x, self.scale, self.zero_point, self.bits, signed
        )

    def quantize_to_int(self, x: Tensor) -> Tensor:
        """Quantiza para inteiro (para inspeção / PTQ real)."""
        assert self.scale is not None
        signed = self.symmetric
        return QuantizationMath.quantize(
            x, self.scale, self.zero_point, self.bits, signed
        )

    def quant_error(self, x: Tensor) -> float:
        """Erro médio quadrático de quantização (MSE)."""
        x_hat = self.quantize_dequantize(x)
        return (x - x_hat).pow(2).mean().item()


class PerChannelQuantizer:
    """
    Quantizador per-channel: um (scale, zero_point) por canal de saída.
    Reduz o erro de quantização em pesos com distribuições heterogêneas.

    shape do peso: [out_features, in_features]
    """

    def __init__(self, bits: int, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.scales: Optional[Tensor] = None
        self.zero_points: Optional[Tensor] = None

    def calibrate(self, w: Tensor):
        """
        Calcula scale/zero_point para cada canal de saída.
        w: [out_channels, ...]
        """
        out_ch = w.shape[0]
        w_flat = w.detach().view(out_ch, -1)   # [out_ch, rest]

        scales = []
        zps = []
        for ch in range(out_ch):
            row = w_flat[ch]
            if self.symmetric:
                s, z = QuantizationMath.compute_scale_symmetric(
                    row.abs().max().item(), self.bits
                )
            else:
                s, z = QuantizationMath.compute_scale_zp_asymmetric(
                    row.min().item(), row.max().item(), self.bits
                )
            scales.append(s)
            zps.append(z)

        self.scales = torch.tensor(scales, dtype=w.dtype, device=w.device)
        self.zero_points = torch.tensor(zps, dtype=torch.int32, device=w.device)

    def quantize_dequantize(self, w: Tensor) -> Tensor:
        """Fake-quantization per-channel."""
        assert self.scales is not None, "Chame .calibrate() antes de quantizar."
        out_ch = w.shape[0]
        w_flat = w.view(out_ch, -1)
        w_out = torch.empty_like(w_flat)

        signed = self.symmetric
        for ch in range(out_ch):
            s = self.scales[ch].item()
            z = self.zero_points[ch].item()
            w_out[ch] = QuantizationMath.fake_quantize(
                w_flat[ch], s, z, self.bits, signed
            )
        return w_out.view_as(w)

    def quant_error(self, w: Tensor) -> float:
        w_hat = self.quantize_dequantize(w)
        return (w - w_hat).pow(2).mean().item()


# ═══════════════════════════════════════════════════════════════
# 5. STRAIGHT-THROUGH ESTIMATOR (STE)  —  gradiente para QAT
# ═══════════════════════════════════════════════════════════════

class FakeQuantizeSTE(torch.autograd.Function):
    """
    Fake quantization com STE no backward.

    Forward : aplica quantização simulada (float → quantizar → dequantizar)
    Backward: passa o gradiente direto (∂L/∂x̂ ≈ ∂L/∂x)

    O STE é necessário porque round() tem derivada zero quase em todo lugar.
    Sem ele, o gradiente não se propaga e o modelo não treina.

    Referência: Bengio et al. "Estimating or Propagating Gradients Through
                Stochastic Neurons for Conditional Computation", 2013.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        scale: float,
        zero_point: int,
        bits: int,
        signed: bool,
    ) -> Tensor:
        ctx.save_for_backward(x)
        ctx.scale = scale
        ctx.zero_point = zero_point
        ctx.bits = bits
        ctx.signed = signed
        return QuantizationMath.fake_quantize(x, scale, zero_point, bits, signed)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        STE: ∂L/∂x = ∂L/∂x̂  (passa gradiente sem modificação)
        Gradientes para scale, zero_point, bits, signed são None.
        """
        return grad_output, None, None, None, None


def fake_quantize_ste(
    x: Tensor,
    scale: float,
    zero_point: int,
    bits: int,
    signed: bool,
) -> Tensor:
    """Wrapper funcional para FakeQuantizeSTE."""
    return FakeQuantizeSTE.apply(x, scale, zero_point, bits, signed)


# ═══════════════════════════════════════════════════════════════
# 6. CAMADAS QUANTIZADAS
# ═══════════════════════════════════════════════════════════════

class QuantizedLinear(nn.Module):
    """
    Camada Linear com suporte a PTQ e QAT, per-tensor e per-channel.

    Modo PTQ  : pesos quantizados na inferência (calibrate_ptq → freeze_weights)
    Modo QAT  : fake-quantization no forward com STE no backward
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        qat: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.qat_mode = qat

        # Parâmetros treináveis
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Quantizador de pesos
        if per_channel:
            self.weight_quantizer = PerChannelQuantizer(bits, symmetric)
        else:
            self.weight_quantizer = TensorQuantizer(bits, symmetric)

        # Quantizador de ativações (per-tensor, assimétrico — ativações são ≥0 após ReLU)
        self.act_quantizer = TensorQuantizer(bits, symmetric=False)

        # Flag: pesos PTQ congelados?
        self._ptq_frozen = False
        self._frozen_weight: Optional[Tensor] = None   # peso fake-quantizado congelado

    # ─── Calibração PTQ ───────────────────────────────────────

    def calibrate_ptq_weight(self):
        """Calibra quantizador de pesos para PTQ."""
        self.weight_quantizer.calibrate(self.weight.data)

    def calibrate_ptq_activation(self, x: Tensor):
        """Calibra quantizador de ativações com dado de calibração."""
        self.act_quantizer.calibrate(x)

    def freeze_ptq(self):
        """
        Congela pesos quantizados: aplica fake-quant uma vez e armazena.
        No forward, usa o peso congelado (mais rápido).
        """
        self.weight_quantizer.calibrate(self.weight.data)
        self._frozen_weight = self.weight_quantizer.quantize_dequantize(
            self.weight.data
        ).clone()
        self._ptq_frozen = True

    # ─── Forward ──────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        if self.qat_mode:
            return self._forward_qat(x)
        elif self._ptq_frozen:
            return self._forward_ptq_frozen(x)
        else:
            return nn.functional.linear(x, self.weight, self.bias)

    def _forward_qat(self, x: Tensor) -> Tensor:
        """
        QAT forward:
          1. Fake-quant de ativação (entrada)
          2. Fake-quant de peso
          3. Linear com pesos/ativações "ruidosos"
        Gradientes fluem via STE.
        """
        # Calibra dinamicamente durante o treino
        self.act_quantizer.calibrate(x.detach())
        self.weight_quantizer.calibrate(self.weight.detach())

        # Fake-quant com STE
        x_q   = fake_quantize_ste(
            x, self.act_quantizer.scale, self.act_quantizer.zero_point,
            self.bits, signed=False
        )
        w_q   = fake_quantize_ste(
            self.weight, self.weight_quantizer.scale,
            self.weight_quantizer.zero_point,
            self.bits, signed=self.symmetric
        ) if not self.per_channel else self._fake_quant_weight_per_channel_ste()

        return nn.functional.linear(x_q, w_q, self.bias)

    def _fake_quant_weight_per_channel_ste(self) -> Tensor:
        """Aplica fake-quant per-channel ao peso com STE."""
        out_ch = self.weight.shape[0]
        w_flat = self.weight.view(out_ch, -1)
        rows = []
        for ch in range(out_ch):
            s = self.weight_quantizer.scales[ch].item()
            z = self.weight_quantizer.zero_points[ch].item()
            rows.append(
                fake_quantize_ste(w_flat[ch], s, z, self.bits, self.symmetric)
            )
        return torch.stack(rows).view_as(self.weight)

    def _forward_ptq_frozen(self, x: Tensor) -> Tensor:
        """Inferência PTQ: peso congelado, ativação fake-quantizada."""
        x_q = self.act_quantizer.quantize_dequantize(x)
        return nn.functional.linear(x_q, self._frozen_weight, self.bias)

    # ─── Utilitários ──────────────────────────────────────────

    def quant_error_weight(self) -> float:
        return self.weight_quantizer.quant_error(self.weight.data)

    def quant_error_activation(self, x: Tensor) -> float:
        return self.act_quantizer.quant_error(x)

    def extra_repr(self) -> str:
        mode = "QAT" if self.qat_mode else ("PTQ-frozen" if self._ptq_frozen else "FP32")
        gran = "per-channel" if self.per_channel else "per-tensor"
        sym  = "symmetric" if self.symmetric else "asymmetric"
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.bits}, {gran}, {sym}, mode={mode}"
        )


# ═══════════════════════════════════════════════════════════════
# 7. MLP  (base + versão quantizável)
# ═══════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """MLP original (FP32)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        return self.linear2(self.relu(self.linear1(x)))


class QuantizedMLP(nn.Module):
    """
    MLP com camadas QuantizedLinear.

    Suporta PTQ e QAT com configuração flexível.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        qat: bool = False,
    ):
        super().__init__()
        self.linear1 = QuantizedLinear(
            input_dim, hidden_dim,
            bits=bits, symmetric=symmetric,
            per_channel=per_channel, qat=qat
        )
        self.relu    = nn.ReLU()
        self.linear2 = QuantizedLinear(
            hidden_dim, output_dim,
            bits=bits, symmetric=symmetric,
            per_channel=per_channel, qat=qat
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        return self.linear2(self.relu(self.linear1(x)))

    # ─── PTQ ──────────────────────────────────────────────────

    def calibrate_ptq(self, calibration_loader, device, n_batches: int = 10):
        """
        Passo 1 do PTQ: roda dados de calibração e observa estatísticas.
        """
        print("  [PTQ] Calibrando...")
        self.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(calibration_loader):
                if i >= n_batches:
                    break
                data = data.to(device)
                x = torch.flatten(data, 1)

                # Calibra ativação de entrada da linear1
                self.linear1.calibrate_ptq_activation(x)

                # Passa pela camada para obter saída intermediária
                out1 = self.relu(
                    nn.functional.linear(x, self.linear1.weight, self.linear1.bias)
                )
                # Calibra ativação de entrada da linear2
                self.linear2.calibrate_ptq_activation(out1)

        # Pesos calibrados e congelados
        self.linear1.freeze_ptq()
        self.linear2.freeze_ptq()
        print("  [PTQ] Calibração concluída. Pesos congelados.")

    def print_quant_errors(self, sample_input: Tensor):
        """Imprime erros de quantização de pesos e ativações."""
        with torch.no_grad():
            x = torch.flatten(sample_input, 1)
            print(f"\n  Erro de quantização (MSE):")
            print(f"    linear1 peso  : {self.linear1.quant_error_weight():.6f}")
            print(f"    linear1 ativ  : {self.linear1.quant_error_activation(x):.6f}")
            out1 = self.relu(
                nn.functional.linear(x, self.linear1.weight, self.linear1.bias)
            )
            print(f"    linear2 peso  : {self.linear2.quant_error_weight():.6f}")
            print(f"    linear2 ativ  : {self.linear2.quant_error_activation(out1):.6f}")


# ═══════════════════════════════════════════════════════════════
# 8. GERENCIADOR DE DADOS  (do código original)
# ═══════════════════════════════════════════════════════════════

class DataloaderManager:
    CONFIGS = {
        "cifar10": {
            "size": 32, "padding": 4, "num_classes": 10, "input_dim": 3 * 32 * 32,
            "stats": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        },
        "cifar100": {
            "size": 32, "padding": 4, "num_classes": 100, "input_dim": 3 * 32 * 32,
            "stats": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        },
        "tiny_imagenet": {
            "size": 64, "padding": 8, "num_classes": 200, "input_dim": 3 * 64 * 64,
            "stats": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        },
        "mnist": {
            "size": 28, "padding": 0, "num_classes": 10, "input_dim": 1 * 28 * 28,
            "stats": ((0.1307,), (0.3081,))
        },
    }

    def __init__(self, root_dir: str, dataset: DatasetName):
        self.root    = Path(root_dir)
        self.dataset = dataset
        self.cfg     = self.CONFIGS[dataset]

    def _get_transforms(self, train: bool):
        mean, std = self.cfg["stats"]
        tf = []
        if train:
            if self.dataset != "mnist":
                tf += [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(self.cfg["size"], padding=self.cfg["padding"]),
                ]
            else:
                tf.append(transforms.RandomRotation(10))
        tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(tf)

    def _get_dataset_instance(self, train: bool):
        tf = self._get_transforms(train)
        if self.dataset.startswith("cifar"):
            cls = datasets.CIFAR10 if self.dataset == "cifar10" else datasets.CIFAR100
            return cls(self.root, train=train, download=True, transform=tf)
        if self.dataset == "mnist":
            return datasets.MNIST(self.root, train=train, download=True, transform=tf)
        path = self.root / "tiny-imagenet-200" / ("train" if train else "val")
        return datasets.ImageFolder(str(path), transform=tf)

    def get_loaders(self, batch_size: int, num_workers: int = 2):
        train_ds = self._get_dataset_instance(train=True)
        val_ds   = self._get_dataset_instance(train=False)
        args = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
        train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **args)
        val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **args)
        print(f"  Dataset: {self.dataset} | Train: {len(train_ds)} | Val: {len(val_ds)}")
        return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════
# 9. LOOPS DE TREINO
# ═══════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    model_name: str = "model",
    save: bool = True,
) -> nn.Module:
    """Loop de treino genérico (FP32 ou QAT)."""
    print(f"\n{'='*60}")
    print(f"  Treinando: {model_name} | Épocas: {epochs} | LR: {lr}")
    print(f"{'='*60}")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 5
    trigger  = 0
    save_path = f"best_{model_name}.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}", leave=False)

        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / len(train_loader):.4f}")

        acc = validate_model(model, val_loader, device, desc=f"Val {epoch+1}")
        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        print(f"  Época {epoch+1:3d}: loss={avg_loss:.4f}  acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            trigger  = 0
            if save:
                torch.save(model.state_dict(), save_path)
        else:
            trigger += 1
            if trigger >= patience:
                print(f"  Early stopping na época {epoch+1}.")
                break

    if save:
        model.load_state_dict(
            torch.load(save_path, weights_only=True, map_location=device)
        )
    print(f"  Melhor acurácia: {best_acc:.2f}%")
    return model


def train_qat(
    fp32_model: nn.Module,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    bits: int,
    symmetric: bool,
    per_channel: bool,
    model_name: str = "qat_model",
) -> Tuple[nn.Module, float]:
    """
    Fluxo completo de QAT:
      1. Copia pesos do modelo FP32 treinado
      2. Substitui camadas por QuantizedLinear no modo QAT
      3. Fine-tune com fake-quantization + STE
    """
    gran = "per-channel" if per_channel else "per-tensor"
    sym  = "symmetric" if symmetric else "asymmetric"
    print(f"\n  [QAT] {bits}bit | {gran} | {sym}")

    # Cria modelo QAT com mesma arquitetura
    cfg    = fp32_model.linear1.in_features
    hidden = fp32_model.linear1.out_features if hasattr(fp32_model.linear1, "out_features") else \
             fp32_model.linear1.weight.shape[0]
    out    = fp32_model.linear2.weight.shape[0]
    inp    = fp32_model.linear1.weight.shape[1]

    qat_model = QuantizedMLP(
        input_dim=inp, hidden_dim=hidden, output_dim=out,
        bits=bits, symmetric=symmetric, per_channel=per_channel, qat=True
    )

    # Copia pesos treinados (FP32) como ponto de partida
    with torch.no_grad():
        qat_model.linear1.weight.copy_(fp32_model.linear1.weight)
        qat_model.linear1.bias.copy_(fp32_model.linear1.bias)
        qat_model.linear2.weight.copy_(fp32_model.linear2.weight)
        qat_model.linear2.bias.copy_(fp32_model.linear2.bias)

    trained = train_model(
        qat_model, train_loader, val_loader, device,
        epochs=epochs, lr=lr * 0.1,   # LR menor para fine-tune
        model_name=model_name, save=True
    )
    acc = validate_model(trained, val_loader, device)
    return trained, acc


def run_ptq(
    fp32_model: nn.Module,
    train_loader,
    val_loader,
    device,
    bits: int,
    symmetric: bool,
    per_channel: bool,
) -> Tuple[nn.Module, float]:
    """
    Fluxo completo de PTQ:
      1. Cria modelo quantizável
      2. Copia pesos FP32
      3. Calibra com dados de treino
      4. Avalia
    """
    gran = "per-channel" if per_channel else "per-tensor"
    sym  = "symmetric" if symmetric else "asymmetric"
    print(f"\n  [PTQ] {bits}bit | {gran} | {sym}")

    out = fp32_model.linear2.weight.shape[0]
    inp = fp32_model.linear1.weight.shape[1]
    hid = fp32_model.linear1.weight.shape[0]

    ptq_model = QuantizedMLP(
        input_dim=inp, hidden_dim=hid, output_dim=out,
        bits=bits, symmetric=symmetric, per_channel=per_channel, qat=False
    )

    with torch.no_grad():
        ptq_model.linear1.weight.copy_(fp32_model.linear1.weight)
        ptq_model.linear1.bias.copy_(fp32_model.linear1.bias)
        ptq_model.linear2.weight.copy_(fp32_model.linear2.weight)
        ptq_model.linear2.bias.copy_(fp32_model.linear2.bias)

    ptq_model.calibrate_ptq(train_loader, device, n_batches=20)
    ptq_model = ptq_model.to(device)
    acc = validate_model(ptq_model, val_loader, device)
    return ptq_model, acc


# ═══════════════════════════════════════════════════════════════
# 10. VALIDAÇÃO E MÉTRICAS
# ═══════════════════════════════════════════════════════════════

def validate_model(model: nn.Module, loader, device, desc: str = "Val") -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in tqdm(loader, desc=desc, leave=False):
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total   += target.size(0)
    return 100.0 * correct / total


def get_model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024 / 1024


def measure_inference_ms(model: nn.Module, loader, device, n_batches: int = 20) -> float:
    model.eval()
    times = []
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= n_batches:
                break
            data = data.to(device)
            if i < 3:      # warmup
                _ = model(data)
                continue
            t0 = time.perf_counter()
            _  = model(data)
            times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1000 if times else 0.0


# ═══════════════════════════════════════════════════════════════
# 11. COMPARAÇÃO ENTRE ESQUEMAS
# ═══════════════════════════════════════════════════════════════

def compare_schemes(
    fp32_model: nn.Module,
    train_loader,
    val_loader,
    device,
    fp32_acc: float,
    qat_epochs: int = 3,
    qat_lr: float = 1e-3,
):
    """
    Compara múltiplos esquemas de quantização e imprime tabela de resultados.
    """

    results = []

    # ─── Configurações a testar ───────────────────────────────
    configs = [
        # (bits, symmetric, per_channel, method_label)
        (8,  True,  False, "PTQ"),
        (8,  False, False, "PTQ"),
        (8,  True,  True,  "PTQ"),
        (4,  True,  False, "PTQ"),
        (4,  True,  True,  "PTQ"),
        (2,  True,  False, "PTQ"),
        (1,  True,  False, "PTQ"),
        (8,  True,  False, "QAT"),
        (8,  False, False, "QAT"),
        (8,  True,  True,  "QAT"),
        (4,  True,  False, "QAT"),
        (4,  True,  True,  "QAT"),
        (2,  True,  False, "QAT"),
        (1,  True,  False, "QAT"),
    ]

    for bits, sym, per_ch, method in configs:
        gran  = "per-ch" if per_ch else "per-tensor"
        quant = "sym" if sym else "asym"
        label = f"{method} INT{bits} {gran} {quant}"

        try:
            if method == "PTQ":
                model_q, acc = run_ptq(
                    fp32_model, train_loader, val_loader, device,
                    bits=bits, symmetric=sym, per_channel=per_ch
                )
            else:
                model_q, acc = train_qat(
                    fp32_model, train_loader, val_loader, device,
                    epochs=qat_epochs, lr=qat_lr,
                    bits=bits, symmetric=sym, per_channel=per_ch,
                    model_name=f"qat_{bits}b_{'sym' if sym else 'asym'}_{'pc' if per_ch else 'pt'}"
                )

            size_mb   = get_model_size_mb(model_q)
            lat_ms    = measure_inference_ms(model_q, val_loader, device)
            degradation = fp32_acc - acc

            results.append({
                "label"      : label,
                "acc"        : acc,
                "degradation": degradation,
                "size_mb"    : size_mb,
                "lat_ms"     : lat_ms,
            })
        except Exception as e:
            print(f"  ERRO em {label}: {e}")
            results.append({"label": label, "acc": -1, "degradation": -1,
                            "size_mb": -1, "lat_ms": -1})

    # ─── Tabela de resultados ─────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  COMPARAÇÃO DE ESQUEMAS DE QUANTIZAÇÃO")
    print(f"  FP32 baseline: acc={fp32_acc:.2f}%  size={get_model_size_mb(fp32_model):.2f}MB")
    print(f"{'='*80}")
    print(f"  {'Esquema':<35} {'Acc%':>7} {'Degr%':>7} {'MB':>7} {'ms/batch':>9}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
    for r in results:
        print(
            f"  {r['label']:<35} "
            f"{r['acc']:>7.2f} "
            f"{r['degradation']:>7.2f} "
            f"{r['size_mb']:>7.2f} "
            f"{r['lat_ms']:>9.2f}"
        )
    print(f"{'='*80}")
    return results


# ═══════════════════════════════════════════════════════════════
# 12. PONTO DE ENTRADA  —  exemplo de uso completo
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quantização manual PTQ + QAT")
    parser.add_argument("--dataset",    default="mnist",    choices=list(DataloaderManager.CONFIGS))
    parser.add_argument("--root",       default="./data")
    parser.add_argument("--batch",      type=int, default=256)
    parser.add_argument("--epochs",     type=int, default=10,  help="Épocas FP32")
    parser.add_argument("--qat_epochs", type=int, default=3,   help="Épocas QAT fine-tune")
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--hidden",     type=int, default=256)
    parser.add_argument("--workers",    type=int, default=2)
    parser.add_argument("--compare",    action="store_true", help="Roda comparação completa")
    parser.add_argument(
        "--bits", type=int, default=8, choices=[1, 2, 4, 8],
        help="Bits para demo rápida (sem --compare)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    # ─── Dados ────────────────────────────────────────────────
    dm = DataloaderManager(args.root, args.dataset)
    train_loader, val_loader = dm.get_loaders(args.batch, args.workers)
    cfg = dm.cfg

    # ─── 1. Treina FP32 ───────────────────────────────────────
    fp32_model = MLP(cfg["input_dim"], args.hidden, cfg["num_classes"])
    fp32_model = train_model(
        fp32_model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, model_name="fp32"
    )
    fp32_acc  = validate_model(fp32_model, val_loader, device, desc="FP32 final")
    fp32_size = get_model_size_mb(fp32_model)
    fp32_lat  = measure_inference_ms(fp32_model, val_loader, device)
    print(f"\n  FP32: acc={fp32_acc:.2f}%  size={fp32_size:.2f}MB  lat={fp32_lat:.2f}ms")

    if args.compare:
        # ─── Comparação completa de todos os esquemas ─────────
        compare_schemes(
            fp32_model, train_loader, val_loader, device,
            fp32_acc=fp32_acc,
            qat_epochs=args.qat_epochs,
            qat_lr=args.lr,
        )
    else:
        # ─── Demo rápida: PTQ + QAT com bits escolhidos ───────
        bits = args.bits

        print(f"\n{'─'*50}")
        print(f"  DEMO: INT{bits} | per-tensor | simétrico")
        print(f"{'─'*50}")

        # PTQ per-tensor simétrico
        ptq_pt_sym, acc_ptq_pt_sym = run_ptq(
            fp32_model, train_loader, val_loader, device,
            bits=bits, symmetric=True, per_channel=False
        )
        print(f"  PTQ per-tensor sym  : {acc_ptq_pt_sym:.2f}%  (Δ={fp32_acc - acc_ptq_pt_sym:.2f}%)")

        # Erros de quantização
        sample_batch = next(iter(val_loader))[0][:32].to(device)
        ptq_pt_sym.print_quant_errors(sample_batch)

        # PTQ per-tensor assimétrico
        ptq_pt_asym, acc_ptq_pt_asym = run_ptq(
            fp32_model, train_loader, val_loader, device,
            bits=bits, symmetric=False, per_channel=False
        )
        print(f"  PTQ per-tensor asym : {acc_ptq_pt_asym:.2f}%  (Δ={fp32_acc - acc_ptq_pt_asym:.2f}%)")

        # PTQ per-channel simétrico
        ptq_pc_sym, acc_ptq_pc_sym = run_ptq(
            fp32_model, train_loader, val_loader, device,
            bits=bits, symmetric=True, per_channel=True
        )
        print(f"  PTQ per-channel sym : {acc_ptq_pc_sym:.2f}%  (Δ={fp32_acc - acc_ptq_pc_sym:.2f}%)")

        # QAT per-tensor simétrico
        qat_pt_sym, acc_qat_pt_sym = train_qat(
            fp32_model, train_loader, val_loader, device,
            epochs=args.qat_epochs, lr=args.lr,
            bits=bits, symmetric=True, per_channel=False,
            model_name=f"qat_int{bits}_pt_sym"
        )
        print(f"  QAT per-tensor sym  : {acc_qat_pt_sym:.2f}%  (Δ={fp32_acc - acc_qat_pt_sym:.2f}%)")

        # QAT per-tensor assimétrico
        qat_pt_asym, acc_qat_pt_asym = train_qat(
            fp32_model, train_loader, val_loader, device,
            epochs=args.qat_epochs, lr=args.lr,
            bits=bits, symmetric=False, per_channel=False,
            model_name=f"qat_int{bits}_pt_asym"
        )
        print(f"  QAT per-tensor asym : {acc_qat_pt_asym:.2f}%  (Δ={fp32_acc - acc_qat_pt_asym:.2f}%)")

        # QAT per-channel simétrico
        qat_pc_sym, acc_qat_pc_sym = train_qat(
            fp32_model, train_loader, val_loader, device,
            epochs=args.qat_epochs, lr=args.lr,
            bits=bits, symmetric=True, per_channel=True,
            model_name=f"qat_int{bits}_pc_sym"
        )
        print(f"  QAT per-channel sym : {acc_qat_pc_sym:.2f}%  (Δ={fp32_acc - acc_qat_pc_sym:.2f}%)")

        print(f"\n{'='*60}")
        print(f"  RESUMO  (FP32 baseline: {fp32_acc:.2f}%)")
        print(f"{'='*60}")
        rows = [
            ("FP32 baseline",           fp32_acc,         0.0),
            (f"PTQ INT{bits} per-tensor sym",   acc_ptq_pt_sym,   fp32_acc - acc_ptq_pt_sym),
            (f"PTQ INT{bits} per-tensor asym",  acc_ptq_pt_asym,  fp32_acc - acc_ptq_pt_asym),
            (f"PTQ INT{bits} per-channel sym",  acc_ptq_pc_sym,   fp32_acc - acc_ptq_pc_sym),
            (f"QAT INT{bits} per-tensor sym",   acc_qat_pt_sym,   fp32_acc - acc_qat_pt_sym),
            (f"QAT INT{bits} per-tensor asym",  acc_qat_pt_asym,  fp32_acc - acc_qat_pt_asym),
            (f"QAT INT{bits} per-channel sym",  acc_qat_pc_sym,   fp32_acc - acc_qat_pc_sym),
        ]
        print(f"  {'Esquema':<35} {'Acc%':>7} {'Degr%':>7}")
        print(f"  {'-'*35} {'-'*7} {'-'*7}")
        for label, acc, deg in rows:
            print(f"  {label:<35} {acc:>7.2f} {deg:>7.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()