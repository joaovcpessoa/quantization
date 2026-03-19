<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$', '$']],
      displayMath: [['$$', '$$']]
    },
    messageStyle: "none"
  });
</script>

# Código

## Dataloader

### Objetivo

Esta classe centraliza a configuração e o instanciamento de diversos datasets populares (CIFAR, MNIST, Tiny ImageNet). Ela automatiza a aplicação de transformações de dados (Data Augmentation), normalização baseada em estatísticas específicas de cada dataset e a criação de objetos DataLoader do PyTorch de forma consistente.

<details><summary><b>Código</b></summary>

```python
DatasetName = Literal["cifar10", "cifar100", "tiny_imagenet", "mnist"]

class DataloaderManager:
    # Configurações centralizadas: tamanho, classes e estatísticas (média, desvio padrão)
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
            # Augmentation específico para imagens coloridas vs escala de cinza
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
        
        return train_loader, val_loader

```
</details>

### Observações Adicionais

O uso de normalização e transformações geométricas é fundamental para a convergência e generalização do modelo.

A operação de `transforms.Normalize(mean, std)` aplica a seguinte fórmula para cada canal da imagem:

$$x_{norm} = \frac{x - \mu}{\sigma}$$

Onde $\mu$ (mean) e $\sigma$ (std) são pré-calculados sobre o conjunto de treino. Isso garante que os dados de entrada tenham *média zero* e *variância unitária*, o que ajuda a evitar que os gradientes desapareçam ou explodam prematuramente durante o backpropagation.

O código também aplica técnicas distintas baseadas no domínio do dado:

- CIFAR/Tiny: Utiliza `RandomHorizontalFlip` e `RandomCrop`. Isso simula variações de posição e orientação que o modelo encontraria no mundo real.
- MNIST: Não utiliza flips horizontais (já que um "3" invertido não é um caractere válido), mas utiliza `RandomRotation`, já que a inclinação da escrita varia entre pessoas.

O parâmetro `pin_memory=True` é uma otimização para treinamento em GPU. Ele aloca as amostras em memória "paginada" (pinned memory), o que permite uma transferência de dados muito mais rápida do Host (CPU) para o Device (GPU) via barramento PCIe. O que pode ser interessante para o treinamento, mas não muito útil para nossa análise voltada para microcontroladores, então talvez eu remova isso futuramente.

## Quantization Range

### Objetivo

Esse código implementa uma função utilitária para quantização cuja função é calcular o intervalo inteiro representável ($q_{min}$, $q_{max}$) dado um número de bits e se a representação é simétrica ou assimétrica.

<details><summary><b>Código</b></summary>

```python
BitWidth = Literal[1, 2, 4, 8] # Permite somente esses valores

def _int_range(bits: int, signed: bool = True) -> Tuple[int, int]:
    if signed and bits == 1:             # Condição especial para binário
        return -1, 1
    if signed:                           # Cálculo para bits assinados (ex: -127,127)      
        q_max = 2 ** (bits - 1) - 1
        q_min = -(2 ** (bits - 1)) + 1
    else:                                # Cálculo para bits não assinados (ex: 0,255)
        q_min = 0
        q_max = 2 ** bits - 1
    return q_min, q_max # Retorna uma tupla com o intervalo
```
</details>

### Observações adicionais

Um inteiro de 8 bits signed normal possui o intervalor $[−128,127]$ ou $[10000000, 01111111]$, lembrando que:

$|−128| = 128$ e $|127| = 127$

O intervalo não é simétrico e em quantização simétrica queremos:

$q = \text{round}(x/scale)$

Se usarmos -128, criamos dois problemas.

1. Saturação desigual, já que:

$q = clamp(q, qmin, qmax)$

valores negativos têm 1 valor extra disponível, o que iria gerar viés estatístico.

2. Overflow na multiplicação inteira

Em redes neurais quantizadas fazemos $acc += q_w ∗ q_x$ (int8 * int8 → int32 accumulator)

$-128 * -128 = 16384$ / $127 * 127 = 16129$

Os produtos negativos extremos são maiores e isso quebra a simetria.

No hardware, em *complemento a dois* existe um detalhe importante, $abs(-128)$ não cabe em `int8`.

$abs(-128) = 128$

Então operações como $abs(x)$ ou $negate(x)$ podem gerar overflow.

## Quantization Math

### Objetivo

Esta classe implementa o mapeamento linear entre o espaço contínuo dos números reais e o espaço discreto dos inteiros. Ela fornece os métodos para calcular os parâmetros de quantização e realizar as operações de ida (`Quantize`), volta (`Dequantize`) e simulação de erro (`Fake Quantize`).

<details><summary><b>Código</b></summary>

```python
class QuantizationMath:
    @staticmethod
    def compute_scale_zp_asymmetric(x_min: float, x_max: float, bits: int) -> Tuple[float, int]:
        """
        Calcula Scale e Zero Point para mapear [x_min, x_max] em [0, 2^bits - 1].
        """
        q_min, q_max = _int_range(bits, signed=False)
        x_range = float(x_max) - float(x_min)
        if x_range < 1e-8: x_range = 1e-8

        scale = x_range / (q_max - q_min)
        zero_point = int(round(q_min - float(x_min) / scale))
        zero_point = int(max(q_min, min(q_max, zero_point)))
        return scale, zero_point

    @staticmethod
    def compute_scale_symmetric(x_abs_max: float, bits: int) -> Tuple[float, int]:
        """
        Calcula Scale para mapear [-max, max] em [-q_max, q_max], com ZP fixo em 0.
        """
        q_min, q_max = _int_range(bits, signed=True)
        if x_abs_max < 1e-8: x_abs_max = 1e-8
        
        scale = float(x_abs_max) / q_max
        return scale, 0

    @staticmethod
    def quantize(x: Tensor, scale: float, zero_point: int, bits: int, signed: bool) -> Tensor:
        """
        Transforma FP32 -> INT: Q(x) = clamp(round(x/S + Z), q_min, q_max)
        """
        q_min, q_max = _int_range(bits, signed=signed)
        x_scaled = x / scale + zero_point
        return torch.clamp(torch.round(x_scaled), q_min, q_max)

    @staticmethod
    def dequantize(x_q: Tensor, scale: float, zero_point: int) -> Tensor:
        """
        Transforma INT -> FP32: x_hat = S * (Q - Z)
        """
        return scale * (x_q - zero_point)

    @staticmethod
    def fake_quantize(x: Tensor, scale: float, zero_point: int, bits: int, signed: bool) -> Tensor:
        """
        Simula o erro de quantização mantendo o tensor em FP32.
        """
        x_q = QuantizationMath.quantize(x, scale, zero_point, bits, signed)
        return QuantizationMath.dequantize(x_q, scale, zero_point)

```
</details>

### Observações Adicionais

Na quantização assimétrica, usamos uma transformação afim para garantir que o valor real $0.0$ seja representado exatamente por um inteiro. Isso é crucial para camadas de *padding* ou funções de ativação como *ReLU*. Usar `uint` permite que você use todos os 256 valores para representar a parte positiva, sem "desperdiçar" metade do range com números negativos que não existem nos seus dados originais. Se $q$ for uint8, a lógica de clipping e manipulação de memória fica muito mais intuitiva para processadores digitais. A relação fundamental é:

$$x = S(q - Z)$$

Onde:
- $S$: Um número real positivo que define o "tamanho" de cada degrau do degrau inteiro.
- $Z$: O valor inteiro que corresponde ao $0.0$ no domínio real.

Durante o *Quantization Aware Training (QAT)*, não podemos treinar diretamente com valores inteiros porque a função de arredondamento tem derivada zero em quase todos os pontos, o que impediria a ocorrência de *Backpropagation*. Por isso fazemos a quantização simulada, cujos passos são:

1. Aplicar o arredondamento e o *clamp*, introduzindo o erro de discretização.
2. Retornar o valor para float.
3. Permitir que o modelo aprenda pesos que sejam robustos à perda de precisão, simulando como o modelo se comportará após a conversão final para ponto fixo.

Outra questão importante, note a verificação `if x_range < 1e-8`. Se o range dos dados for zero (ex: uma camada cujos pesos são todos iguais), a escala tenderia ao infinito ou causaria uma divisão por zero. O *epsilon* garante que o cálculo permaneça estável.

Essa verificação de divisão por zero é o que chamamos de "programação defensiva". No contexto de quantização, ela evita que seu código quebre em cenários matematicamente impossíveis ou em dados "mortos". Imagine que você está passando os dados de uma camada de uma rede neural e, por algum motivo, como por exemplo a morte de um neurônio após uma ativação ReLU, todos os valores de entrada são exatamente iguais a `0.5`.

Nesse caso: $x_{max} = 0.5$ e $x_{min} = 0.5$

Logo: $x_{range} = 0.5 - 0.5 = 0$

Se você tentar calcular o `scale` sem a trava:

$$S = \frac{0}{q_{max} - q_{min}} = 0$$

Até aí, tudo bem, mas o problema acontece no cálculo do **Zero Point**:

$$Z = q_{min} - \frac{x_{min}}{S} \longrightarrow Z = q_{min} - \frac{0.5}{0}$$

Você terá um erro de `ZeroDivisionError` no Python ou um `NaN` (Not a Number) em C++/CUDA, o que invalidaria todo o treinamento ou inferência da sua rede.

Esse valor ($1 \times 10^{-8}$) é o que chamamos de **epsilon** ($\epsilon$). É um número pequeno o suficiente para não distorcer a matemática da rede, mas grande o suficiente para manter a estabilidade numérica. Ao forçar `x_range = 1e-8`, o seu `scale` será um número extremamente pequeno, mas válido. Isso garante que o `zero_point` possa ser calculado e que o processo de quantização continue sem travar o sistema.

Por incrível que pareça, isso pode acontecer em algumas situações, por exemplo:

- Se uma região inteira da imagem for negativa, a ReLU zera tudo. O $x_{min}$ e $x_{max}$ serão ambos $0$.
- Em modelos com *Sparsity* (poda de neurônios), é comum encontrar tensores cheios de zeros.
- Se você passar apenas um valor para a função em vez de uma lista de valores.

## Quantization Observers

### Objetivo

Estes módulos monitoram o fluxo de dados das ativações e pesos para registrar os valores mínimos e máximos globais ou específicos por canal. Essas estatísticas são a base para o cálculo do $scale$ e *zero point* durante a calibração em PTQ ou no treinamento em QAT. Servindo para ambos.

<details><summary><b>Código</b></summary>

```python
class MinMaxObserver:
    """
    Captura os valores globais de mínimo e máximo de um tensor ao longo do tempo.
    Ideal para Ativações, onde se busca um range único para todo o mapa de características.
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
        self.min_val, self.max_val = None, None

class PerChannelObserver:
    """
    Captura o mínimo e máximo individualmente para cada canal de saída.
    x shape: [out_features, in_features] ou [out_channels, in_channels, H, W]
    Muito utilizado para Pesos (Weights), permitindo maior precisão granular.
    """
    def __init__(self):
        self.min_vals: Optional[Tensor] = None
        self.max_vals: Optional[Tensor] = None

    def update(self, x: Tensor):
        # min/max ao longo da dimensão de saída (geralmente dim=0 ou dim=1 dependendo da camada)
        x_min = x.detach().min(dim=1).values
        x_max = x.detach().max(dim=1).values
        if self.min_vals is None:
            self.min_vals = x_min
            self.max_vals = x_max
        else:
            self.min_vals = torch.min(self.min_vals, x_min)
            self.max_vals = torch.max(self.max_vals, x_max)

    def reset(self):
        self.min_vals, self.max_vals = None, None

```
</details>

### Observações Adicionais

Em redes neurais, os pesos de diferentes neurônios, que chamamos de canais, podem ter distribuições de magnitude drasticamente diferentes. Se usarmos um único `MinMaxObserver` global para todos os canais, um canal com valores altos esticaria o range de quantização e canais com valores pequenos sofreriam um erro de discretização enorme. Por isso criei também `PerChannelObserver`, onde cada canal de saída tem sua própria escala, preservando a precisão individual de cada filtro da convolução ou neurônio da camada linear.

Importante também é o uso de `.detach()`. A observação é um processo de monitoramento estatístico e não deve fazer parte do grafo de computação para o backpropagation. O objetivo é apenas coletar metadados sobre a dinâmica dos valores.

Embora a quantização por canal seja excelente para acurácia, ela exige que o hardware suporte a aplicação de múltiplos *scales* na acumulação final.

- Weights: Quase sempre quantizados *Per-Channel*.
- Activations: Quase sempre quantizadas *Per-Tensor*, devido ao custo computacional de gerenciar escalas diferentes durante o cálculo da ativação em tempo real.

## Quantizers

### Objetivo

Estas classes encapsulam o processo de calibração e transformação de dados. Enquanto o `TensorQuantizer` trata o tensor como uma unidade única, ideal para ativações, o `PerChannelQuantizer` aplica parâmetros independentes para cada canal de saída, essencial para pesos de camadas convolucionais e lineares, maximizando a fidelidade do sinal.

<details><summary><b>Código</b></summary>

```python
class TensorQuantizer: # Define uma classe chamada TensorQuantizer
    """Aplica um único par de scale e zero_point para o tensor inteiro"""
    def __init__(self, bits: int, symmetric: bool = True): # Construtor da classe
        self.bits = bits                      # Guarda o número de bits
        self.symmetric = symmetric            # Guarda se a quantização é simétrica
        self.scale: Optional[float] = None    # Define scale (float/None)
        self.zero_point: Optional[int] = None # Define zp (int/None)

    def calibrate(self, x: Tensor):
        """Calcula parâmetros baseados no min/max do tensor"""
        if self.symmetric:
            # x.detach() -> remove o tensor do grafo de gradiente (não participa de backprop)
            # .abs() -> calcula o valor absoluto
            # .max() -> pega o valor máximo
            # .item() -> converte tensor → número Python 
            abs_max = x.detach().abs().max().item()
            # calcula scale e zp simétrico
            self.scale, self.zero_point = QuantizationMath.compute_scale_symmetric(abs_max, self.bits)
        else:
            # Pega o valor mínimo e máximo do tensor
            x_min = 
            x_max = x.detach().min().item(), x.detach().max().item()
            # calcula scale e zp assimétrico
            self.scale, self.zero_point = QuantizationMath.compute_scale_zp_asymmetric(x_min, x_max, self.bits)

    def quantize_dequantize(self, x: Tensor) -> Tensor:
        """Fake-quantization: simula o erro sem sair do domínio float"""
        # Verifica se calibrate() foi chamado antes, caso contrário, dá erro
        assert self.scale is not None, "Chame .calibrate() antes de quantizar"
        signed = self.symmetric # Define o tipo de inteiro
        # Aplica a quantização simulada
        return QuantizationMath.fake_quantize(x, self.scale, self.zero_point, self.bits, signed)

    def quantize_to_int(self, x: Tensor) -> Tensor:
        """Quantiza para inteiro (para inspeção/PTQ real)"""
        assert self.scale is not None
        signed = self.symmetric # Define o tipo de inteiro (signed → [-128,127], unsigned → [0,255])
        # Aplica a quantização real para inteiros
        return QuantizationMath.quantize(x, self.scale, self.zero_point, self.bits, signed)

    def quant_error(self, x: Tensor) -> float:
        """Calcula o Mean Squared Error (MSE) da quantização"""
        x_hat = self.quantize_dequantize(x)     # Aplica fake quantization
        return (x - x_hat).pow(2).mean().item() # Erro médio quadrático

class PerChannelQuantizer:
    """Calcula (scale, zero_point) individualmente por canal (dim=0)."""
    def __init__(self, bits: int, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.scales: Optional[Tensor] = None
        self.zero_points: Optional[Tensor] = None

    def calibrate(self, w: Tensor):
        out_ch = w.shape[0]
        w_flat = w.detach().view(out_ch, -1)
        scales, zps = [], []

        for ch in range(out_ch):
            row = w_flat[ch]
            if self.symmetric:
                s, z = QuantizationMath.compute_scale_symmetric(row.abs().max().item(), self.bits)
            else:
                s, z = QuantizationMath.compute_scale_zp_asymmetric(row.min().item(), row.max().item(), self.bits)
            scales.append(s); zps.append(z)

        self.scales = torch.tensor(scales, dtype=w.dtype, device=w.device)
        self.zero_points = torch.tensor(zps, dtype=torch.int32, device=w.device)

    def quantize_dequantize(self, w: Tensor) -> Tensor:
        assert self.scales is not None
        out_ch = w.shape[0]
        w_flat = w.view(out_ch, -1)
        w_out = torch.empty_like(w_flat)

        for ch in range(out_ch):
            w_out[ch] = QuantizationMath.fake_quantize(
                w_flat[ch], self.scales[ch].item(), self.zero_points[ch].item(), self.bits, self.symmetric
            )
        return w_out.view_as(w)

```
</details>

### Observações Adicionais

Pesos de redes neurais costumam ter faixas dinâmicas muito variadas entre filtros. Em uma camada com 64 filtros, um único filtro com valores discrepantes (outliers) pode forçar um `scale` muito alto para a camada inteira, "esmagando" a precisão dos outros 63 filtros.

O `PerChannelQuantizer` resolve isso tratando cada linha da matriz de pesos (ou canal da convolução) como um domínio de quantização independente.

O método `quant_error` utiliza o **Erro Médio Quadrático (MSE)**:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2$$

Essa métrica é fundamental para:

* Validar se o `bit-width` escolhido (ex: 4 bits vs 8 bits) é suficiente.
* Comparar se a quantização Simétrica está perdendo muita informação em relação à Assimétrica.

* **Symmetric (Signed):** Simplifica a lógica de hardware, pois o $Zero-Point$ é sempre $0$. Operações de $acc += q_w * q_x$ tornam-se multiplicações simples.
* **Asymmetric (Unsigned):** Mais preciso para distribuições que não são centradas no zero (como ativações após ReLU), mas exige hardware que suporte a compensação do $Zero-Point$ durante a aritmética.

Você gostaria que eu criasse um script de visualização (usando Matplotlib) para plotar o histograma de um tensor original vs. o tensor quantizado por essas classes? Seria uma forma excelente de ver o "esmagamento" dos bins na prática.

#### PerChannelObserver

Fiz uma versão para granularidade por canal. Defini uma classe para observar mínimos e máximos por canal de saída.
Agora cada canal, representados por linha da matriz, terá seu próprio `min/max`.

Para camada linear o peso tem shape `[out_features, in_features]`, então cada linha corresponde a um neurônio de saída

```python
    def __init__(self):
        self.min_vals: Optional[Tensor] = None
        self.max_vals: Optional[Tensor] = None

    def update(self, x: Tensor):
        """x shape: [out_features, in_features]"""
        x_min = x.detach().min(dim=1).values
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
```
Agora temos vetores para armazenar o mínimo e máximo por canal

A função `update` recebe um tensor 2D com os pesos da camada. A parte mais importante é:

- `dim=0` → índice da linha → neurônio de saída
- `dim=1` → índice da coluna → feature de entrada

Queremos um min/max por neurônio de saída, ou seja, um vetor do tamanho de `[out_features]`.

```python
x_min = x.min(dim=1).values
```

Reduz ao longo das colunas e mantém a dimensão das linhas.

- x.shape = [out_features, in_features]
- min(dim=1) → reduz in_features
- resultado → [out_features]

Para cada linha (neurônio), pega o menor valor daquela linha.

Exemplo:

```text
x = [[1, 2, 3],
     [4, 0, 6]]
x_min = [1, 0]
```

Assim como no caso por tensor, usamos o mesmo raciocío para salvar os valores. Se não há valores, salvamos, se já temos valores anteriores, comparamos e atualizamos elemento por elemento. Ao final resetamos os valores para a próxima calibração.

Exemplo: 

Suponha que já tínhamos:

```python
min_vals = [-0.8, -0.3]
max_vals = [ 0.9,  0.5]
```

Agora chega um novo batch de pesos:

```python
x_min = [-1.2, -0.1]
x_max = [ 0.7,  0.8]
```

Atualização:

```python
novo_min = min([-0.8, -0.3], [-1.2, -0.1]) # = [-1.2, -0.3]
novo_max = max([0.9, 0.5], [0.7, 0.8])     # = [0.9, 0.8]
```

Cada canal é atualizado independentemente. Isso é element-wise.

#### É possível fazer por linha e por coluna?

Sim, pode ser por coluna, mas qual dimensão usar depende da semântica do tensor.

Vamos usar o nosso caso, uma camada `Linear`. Para uma camada `Linear` no PyTorch:

```python
weight.shape = [out_features, in_features]
```

Ou seja, linhas (dim=0) são neurônios de saída e colunas (dim=1) são as entradas de cada neurônio.

Como cada linha corresponde a um neurônio diferente, se você faz:

```python
x_min = x.min(dim=1).values
```

Estamos calculando `Min/max` dos pesos que alimentam cada neurônio de saída.
Isso significa que cada neurônio terá sua própria escala de quantização.

Isso faz sentido porque cada neurônio pode ter distribuição diferente de pesos, então acaba por melhorar a precisão

Se fizermos:

```python
x_min = x.min(dim=0).values
```

Agora estamos calculando min/max por coluna. Isso significa que cada feature de entrada terá sua própria escala.

Conceitualmente isso quer dizer que todos os neurônios compartilharão a mesma escala para aquela feature, então estaríamos agrupando pesos pelo eixo da entrada. Na multiplicação, cada linha de `W` produz um valor de saída

```python
y = W x
y_i = W_i · x
```

Então temos o seguinte:
- Cada linha é um "filtro"
- Cada filtro pode ter escala própria

Faz sentido quantizar cada filtro separadamente

E no caso de uma Convolução?

Para `Conv2d`:

```
weight.shape = [out_channels, in_channels, kH, kW]
```