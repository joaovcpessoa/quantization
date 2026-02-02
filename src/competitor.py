import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        # Depthwise
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        # Pointwise
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=200):
        super(MobileNetV1, self). __init__()
        self.quant = QuantStub()
        
        self.model = nn.Sequential(
            conv_bn(3, 32, 2), # Camada 0
            conv_dw(32, 64, 1), # Camada 1
            conv_dw(64, 128, 2), # Camada 2
            conv_dw(128, 128, 1), # Camada 3
            conv_dw(128, 256, 2), # Camada 4 -> Early Exit aqui
        )
        
        # Ramo de Saída Antecipada
        self.early_exit = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
        self.after_exit = nn.Sequential(
            conv_dw(256, 256, 1),
            # ... adicione as demais camadas conforme o padrão V1 ...
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(256, num_classes) # Simplificado para o exemplo
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        
        # Saída Antecipada
        out_early = self.early_exit(x)
        out_early = self.dequant(out_early)
        
        # Caminho Principal
        x = self.after_exit(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        
        return x, out_early
    
import torch.ao.quantization as quant

def prepare_experiment(model, mode="per_channel", fold_bn=True):
    model.train() # Necessário para preparação
    
    # 1. Aplicar BN Folding (Fusão)
    if fold_bn:
        # Exemplo de fusão para o primeiro bloco
        # Em um modelo real, você faria um loop por todos os submodulos
        torch.ao.quantization.fuse_modules(model.model[0], ['0', '1', '2'], inplace=True)
    
    # 2. Configurar Observadores (Per-Channel vs Per-Tensor)
    if mode == "per_channel":
        qconfig = quant.get_default_qconfig('fbgemm')
    else: # per_tensor
        qconfig = quant.QConfig(
            activation=quant.MinMaxObserver.with_args(reduce_range=True),
            weight=quant.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        )
    
    model.qconfig = qconfig
    quant.prepare(model, inplace=True)
    return model

# Caso 1: Per-channel + Fold BN (O mais próximo da NVIDIA)
model_pc_fold = MobileNetV1()
model_pc_fold = prepare_experiment(model_pc_fold, mode="per_channel", fold_bn=True)
# ... Calibração com dados ...
model_int8 = quant.convert(model_pc_fold)

# Caso 2: Per-tensor (Sem Fold)
model_pt = MobileNetV1()
model_pt = prepare_experiment(model_pt, mode="per_tensor", fold_bn=False)
# ... Calibração e Conversão ...