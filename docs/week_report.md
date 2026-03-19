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

# Relatório

## Ato 0

### Escolha do hardware

Escolher o hardware é importante pois a performance final será determinada pelo backend de quantização do hardware escolhido. O backend, nesse contexto, é o "motor" que realmente executa os cálculos matemáticos no hardware. Sem um backend específico, o processador tentaria rodar números inteiros usando as mesmas instruções de ponto flutuante, o que não traria ganho nenhum de velocidade. O backend traduz as operações da sua rede neural para instruções de baixo nível que o seu processador entende de forma otimizada.

As principais funções são:

- Aceleração de Hardware: Utiliza instruções especiais do processador (como AVX-512 em Intel ou NEON em chips ARM) para processar vários dados de uma vez (SIMD)
- Gerenciamento de Memória: Organiza como os pesos da rede são carregados no cache para evitar gargalos

### Quando um modelo é considerado "Integer Only"?

Um modelo é considerado 'integer-only' quando todas as suas operações computacionais (multiplicações, adições, ativações, etc.) são realizadas usando aritmética de inteiros de baixa precisão. Isso significa que não há conversões intermediárias para ponto flutuante durante a inferência. O fluxo de dados é puramente de inteiros do início ao fim do modelo.

O backend `fbgemm` do PyTorch (para CPUs) e o `TensorRT` da NVIDIA são capazes de executar modelos de forma totalmente quantizada para certas arquiteturas e operações.

A quantização híbrida (simulada/mixed-precision) é uma abordagem mais comum e flexível, especialmente durante o processo de desenvolvimento e calibração. Neste cenário, algumas operações são realizadas em inteiros, mas conversões intermediárias ainda podem ocorrer em ponto flutuante.

Embora isso possa parecer flexível, já que permite quantizar apenas as partes mais intensivas computacionalmente do modelo, mantendo outras partes em FP32 para preservar a acurácia ou simplificar a implementação, não funcionria em hardware embarcado que não possui suporte para FP32.

Com base na estrutura do exemplo, dá para perceber que é um exemplo de quantização 'híbrida'. O PyTorch adota a abordagem híbrida por padrão em PTQ por várias razões:

- Facilidade de Uso: Simplifica a integração com o ecossistema PyTorch existente, onde muitas operações e o treinamento são em FP32.

- Compatibilidade: Garante que o modelo possa interagir com outras partes do código que esperam entradas/saídas em FP32.

- Acurácia: Minimiza a perda de acurácia ao permitir que operações sensíveis (ou não otimizadas para inteiros) permaneçam em FP32 ou sejam dequantizadas temporariamente.

- Backends: A capacidade de executar um modelo de forma totalmente 'integer-only' depende do backend de quantização (e.g., fbgemm, qnnpack) e do hardware subjacente. O PyTorch tenta otimizar para o backend disponível, mas a estrutura com QuantStub/DeQuantStub é a representação padrão para a quantização simulada.

Para obter um modelo verdadeiramente 'integer-only' no PyTorch, você precisaria garantir que:

1. Todas as operações são suportadas: Cada operação no seu grafo computacional tem uma implementação otimizada para inteiros no backend de quantização escolhido.
2. Fusão de Operações: Operações como Conv + ReLU ou Linear + ReLU são fundidas em uma única operação quantizada para evitar dequantizações intermediárias.
3. Remoção de DeQuantStub: A DeQuantStub final seria removida ou posicionada apenas no ponto onde a saída precisa ser consumida por um sistema que espera FP32. Para inferência puramente em hardware de inteiros, a saída pode permanecer em INT8.


## Ato I

Para inferência de modelos na AWS, o hardware proprietário de ponta é o AWS Inferentia, que possui os chips Inferentia1 e Inferentia2. Diz a documentação oficial que eles foram projetados especificamente para oferecer alto rendimento e baixa latência, otimizando o custo por inferência em comparação com GPUs tradicionais.

### AWS Inferentia2

O coração das instâncias Inf2 é o acelerador NeuronCore-v2. Diferente de uma CPU de propósito geral, ele é uma arquitetura otimizada para operações tensoriais, já que cada chip possui núcleos especializados que executam operações de álgebra linear de alta densidade, utiliza memória de alta largura de banda para evitar gargalos durante o carregamento dos pesos do modelo e ainda permite a comunicação direta entre chips para modelos robustos, como LLMs, reduzindo a dependência da CPU principal.

### Representação Numérica e Aritmética

Assim como quase toda a arquitetura moderna, esse hardware utiliza *two's complement* para representar números inteiros com sinal.

O compilador da AWS, AWS Neuron SDK, permite um controle fino sobre como a precisão é tratada. No contexto de Deep Learning, a maioria das operações ocorre em FP16, BF16 ou INT8.

O que esperamos no final de um produto escalar é truncamento, mas o padrão do Neuron SDK é arredondamento para o valor mais próximo (*round-to-nearest-even*).

### Testando no Ambiente

Para validar o comportamento exato da aritmética de ponto fixo no chip preciso realizar um experimento, para isso é necessário um ambiente com o AWS Neuron SDK instalado. O foco será comparar o comportamento do acelerador NeuronDevice contra o da CPU padrão.

Escrevi um script baseado nas documentações oficiais que utiliza o framework `PyTorch` com a extensão `torch-neuronx`. Vou tentar forçar uma situação onde a diferença entre truncamento e arredondamento seja visível em operações de ponto flutuante e verificar o comportamento de sinal para o complemento a 2.

```python
import torch
import torch_neuronx
import numpy as np

val = 1.7
tensor_cpu = torch.tensor([val], dtype=torch.float32)

class RoundModel(torch.nn.Module):
    def forward(self, x):
        return x.to(torch.int32) # Conversão explícita

model = RoundModel()
neuron_model = torch_neuronx.trace(model, tensor_cpu)
output_cpu = model(tensor_cpu)
output_neuron = neuron_model(tensor_cpu)

print(f'Valor Original:   {val}')
print(f'Resultado CPU:    {output_cpu.item()}')
print(f'Resultado Neuron: {output_neuron.item()}')

neg_val = -1
tensor_neg = torch.tensor([neg_val], dtype=torch.int32)
res_neg = neuron_model(tensor_neg.to(torch.float32))

print(f'Input negativo: {neg_val} -> Output: {res_neg.item()}')
```

Esse teste não valeu de nada na prática. Assim como o teste que fiz em CPU.
O problema é que Python abstrai demais o hardware. Para ver o que de fato ocorre, precisamos explorar em C ou até Assembly.

Minha ideia é forçar um caso onde o resultado exato não cabe em float32, outro para deixar a FPU arredondar e um para observar o resultado nos bits. Quero ver o arredondamento IEEE-754 acontecendo dentro da FPU.

Float32 tem 23 bits de mantissa (~7 dígitos decimais). Depois de certo ponto, somar $1$ não muda mais o número porque a mantissa não consegue representar.

```c
#include <stdio.h>

int main() {
    float x = 16777216.0f;  // 2^24
    float y = 1.0f;

    float result = x + y;

    printf("x = %.0f\n", x);
    printf("x + 1 = %.0f\n", result);
}

// Resultado:
// x = 16777216
// x + 1 = 16777216
// Matematicamente deveria ser: 16777217
// Mas a FPU faz: resultado exato -> arredondamento para float32
```

Agora vamos ver o *bit pattern*.

```c
#include <stdio.h>
#include <stdint.h>

int main() {
    float x = 16777216.0f;
    float y = 1.0f;

    float result = x + y;

    uint32_t *bits = (uint32_t*)&result;

    printf("Resultado: %.0f\n", result);
    printf("Bits: 0x%X\n", *bits);
}

// Resultado:
// Resultado: 16777216
// Bits: 0x4B800000
// Isso mostra o valor já arredondado pela FPU
```

Aqui vamos ver a FPU descartar completamente o $1e-8$ porque não há bits suficientes na mantissa para representar esse incremento mantendo a magnitude do $1.0$.

```c
#include <stdio.h>

int main() {
    float a = 1.0f;
    float b = 1e-8f;

    float c = a + b;

    printf("1.0 + 1e-8 = %.10f\n", c);
}

// Resultado:
// 1.0000000000
// A FPU faz: 1.00000001 -> round-to-nearest -> 1.00000000
```

Vou tentar ler o registrador da FPU.

```c
#include <stdio.h>
#include <fenv.h>

int main() {
    int mode = fegetround();

    if(mode == FE_TONEAREST)
        printf("Round to nearest\n");
}

// Resultado:
// Round to nearest
```

Agora vou tentar trocar o modo da FPU. Normalmente, quando o resultado de uma operação matemática não pode ser representado exatamente em binário, a CPU usa um padrão chamado 'arredondamento para o par mais próximo', como vimos.

Ao usar a biblioteca <fenv.h> e a função `fesetround` (FE_DOWNWARD), você está forçando a unidade de ponto flutuante do processador a mudar seu comportamento: todo resultado de uma operação deve ser arredondado para o número representável imediatamente abaixo (ou igual) ao valor exato.

A operação 1.9f + 1.0f resulta em 2.9.

O problema é que 2.9 não possui uma representação binária exata (ele é uma dízima periódica em binário).

Sem a alteração, o sistema arredondaria 2.9 para o valor binário mais próximo, que é algo como 2.900000095....

Com FE_DOWNWARD, o sistema ignora o "excesso" e trava no número representável imediatamente anterior a 2.9.

```c
#include <stdio.h>
#include <fenv.h>

int main() {
    fesetround(FE_DOWNWARD);
    float x = 1.9f + 1.0f;
    printf("%f\n", x);
}

// Resultado:
// 2.900000
```

O compilador gera instruções SIMD como:

```
addss
mulss
vfmadd132ps
```

Essas instruções calculam, arredondam e armazenam. O arredondamento acontece na unidade FPU/SIMD do processador. Ocorre dentro da instrução de ponto flutuante, então só podemos observar o resultado final arredondado.

## Ato II

### BF16

Como eu sou mal educado, não apresentei o BF16 para vocês.

BF16, Marcelo e Rafael, Marcelo e Rafael, BF16. Agora que estamos devidamente apresentados, deixa eu caracterizar nosso amigo.

BF16 (bfloat16) é um formato de número em ponto flutuante de 16 bits bastante usado em aprendizado de máquina para acelerar cálculos e reduzir uso de memória. Popularizado pelo Google para uso em hardware de IA como as TPU's (Tensor Processing Units), mas hoje também é suportado por GPUs e CPUs modernas de empresas como NVIDIA, Intel e AMD.

Ele é basicamente uma versão menor do formato FP32, ele é menos preciso, porém mais rápido e leve, o que permite treinar modelos maiores usando menos memória e energia. Um número bf16 usa 16 bits divididos assim:
- 1 bit → sinal
- 8 bits → expoente
- 7 bits → mantissa (precisão)

Comparação rápida:

| Formato | Bits | Expoente | Mantissa |
| ------- | ---- | -------- | -------- |
| FP32    | 32   |	8	    | 23	|
| FP16	  | 16   |	5	    | 10	|
| BF16	  | 16   |	8	    | 7	    |

Esse formato mantém o mesmo tamanho de expoente do FP32, então consegue representar números muito grandes ou muito pequenos.
Usado em treinamento de modelos por frameworks como TensorFlow e Pytorch.

Agora, como diabos isso pode ser mais rápido? Bom, vamos acender uma tocha e explorar esse poço mais profundamente.

Um número em ponto flutuante é representado como 

$$x = (-1)^s \times 1.m \times 2^e$$

onde:
- s -> sinal
- m -> mantissa
- e -> expoente

Na multiplicação os expoentes só são somados, já que é uma operação mais barata computacionalmente falando e o custo real fica em multiplicar as mantissas.

$$ x \cdot y = (1.m_x \times 1.m_y) \times 2^{e_x + e_y} $$

Se a mantissa tem $n$ bits, a multiplicação binária custa aproximadamente $O(n^2)$ para multiplicadores simples usados em hardware.

Então para FP32, temos $23^2 = 529$ e BF16 $7^2 = 49$, então para saber a taxa:

$$ \frac{529}{49} \approx 10.8$$

A multiplicação da mantissa em BF16 pode exigir ~10× menos operações lógicas que em FP32. Isso permite utilizarmos unidades de multiplicação menores, mais multiplicadores por chip e maior efeito de paralelismo.

Como já está claro para todos nós, movimentar dados custa mais energia que calcular. Se pensarmos que FP32 equivale a 4 bytes e BF16 a 2 bytes, então o cache carrega $2\times$ mais valores, a memória move metade dos dados e os barramentos ficam menos congestionados. Isso também acelera operações de matriz como $C = A \times B$ que são o núcleo de redes neurais.

Como os números são menores os registradores podem armazenar mais valores
e as unidades SIMD processam mais números por ciclo. 

Obs.: Eu cheguei a comentar sobre SIMD em algum momento no outro documento, mas se quiserem posso adicionar aqui para tornar a leitura mais clara e sequencial.

Reza a lenda que o treinamento de redes neurais não precisa de uma precisão tão alta, já que gradientes são ruidosos e estocásticos por natureza:

$$ \theta_{t+1} = \theta_t - \eta \nabla L(\theta) $$

Então pequenas imprecisões da mantissa não afetariam muito o resultado final, mas manter 8 bits de expoente evita underflow/overflow durante o treinamento além de todos os outros benefícios que comentei antes.

### Experimentos

**Geração dos dados**

Entrada:

$X \sim N(0,1)$

Pesos e bias são inicializados aleatoriamente.

Dimensões:
- $X \in \mathbb{R}^{10\times5}$
- $W \in \mathbb{R}^{5\times5}$
- $b \in \mathbb{R}^{5}$

**Inferência em FP32**

Nosso querido *baseline*. Todas as operações são realizadas em ponto flutuante. A camada linear:

$Z = X \cdot W + b$

Quantização simétrica signed int8:

```python
scale_x = max(|X|)/127
scale_w = max(|W|)/127
```
O intervalo é $[−127,127]$.

$q=round(x/scale)$

Aplicação da ativação:

$A_{fp32} = \max(0, Z)$

Essa saída é usada como referência de precisão.

**Quantização das ativações**

Os valores de entrada são convertidos para INT8.

Calculamos a escala:

$scale_x = \frac{\max(|X|)}{127}$

Quantizamos:

$X_q = round\left(\frac{X}{scale_x}\right)$

Dequantizamos

$X_{dq} = scale_x \cdot X_q$

**Quantização híbrida**

Aqui as entradas são quantizadas e os pesos permanecem em FP32

$X \rightarrow X_q$

$X_q \rightarrow X_{dq}$

$Z = X_{dq} W + b$

$A = \max(0, Z)$

Isso só server para medir o erro de quantização das ativações.

**Quantização pura**

Tanto entradas quanto pesos são quantizados.

$X_q = round(X / scale_x)$

$W_q = round(W / scale_w)$

$Z_{int32} = X_q W_q$

A multiplicação de INT8 produz acumuladores INT32. Voltamos ao domínio real:

$Z_{fp32} = (scale_x \cdot scale_w) Z_{int32}$

$Z = Z_{fp32} + b$

$A = \max(0, Z)$

Uma outra questão importante é que utilizo quantização simétrica para os pesos e assimétrica para ReLU.

Após ReLU temos: $A=max(0,x)$

Logo: $A \in [0,max]$

O intervalo do uint8 é $[0,255]$. Então se escolhermos

$scale=\frac{max}{255}$	​

Onde o $zp$ é calculado para garantir que o zero real seja representado sem erro:

$$zp = q_{min} - \frac{r_{min}}{scale}$$

No caso da ReLU:
- $r_{min}$ (valor real mínimo): É sempre 0, pois a ReLU corta tudo abaixo disso
- $q_{min}$ (valor inteiro mínimo): Usar uint8 para a saída, cujo valor mínimo é 0

Substituindo na fórmula:

$$zp = 0 - \frac{0}{scale} \implies \mathbf{zp = 0}$$

Dessa forma, nós estamos distribuindo os 256 níveis de densidade numérica apenas na parte positiva. Se tentassemos usar uma quantização simétrica para a ReLU, estariamos "jogando fora" 128 níveis, ou seja, toda a parte negativa do INT8, já que a ReLU nunca produziria valores para preencher esses slots vazios. Usar a abordagem assimétrica simplificada mantém a simplicidade computacional e ganha o dobro de precisão para representar os valores positivos em comparação ao uso de INT8 simétrico.

Lembrando também que uso duas abordagens para converter valores, que são:

- **Arredondamento**: `np.round(x/scale)` arredonda o valor para o inteiro mais próximo. Esta é a abordagem mais comum, pois tende a minimizar o erro médio, distribuindo-o de forma mais equilibrada. Matematicamente, `round(v)` mapeia `v` para `floor(v + 0.5)`.

- **Truncamento**: `np.trunc(x/scale)` remove a parte fracionária, arredondando o valor em direção a zero. Esta operação é, por vezes, computacionalmente mais barata em certos hardwares. No entanto, introduz um viés sistemático, pois todos os valores são movidos na direção de zero, o que geralmente resulta num erro de quantização maior.

**Avaliação**

Para avaliar a qualidade da quantização, não basta olhar apenas para os valores brutos. Existem métricas que revelam se estamos perdendo a direção dos dados ou apenas adicionando um ruído aceitável.

Selecionei alguns erros que podem ser interessantes de serem observados.

**SQNR (Signal-to-Quantization-Noise Ratio)**

Métrica de ouro em processamento de sinais que mede a potência do sinal em relação à potência do erro, medido em dB.
Diferente do MSE, o SQNR é relativo. Um erro de 0.1 é irrelevante se os dados variam até 100, mas crítico se variam até 0.2.
Nesse caso, quanto maior, melhor.

$$SQNR = 10 \cdot \log_{10}\left(\frac{\sum A_{fp32}^2}{\sum (A_{fp32} - A_{pure})^2}\right)$$

**Erro Máximo Absoluto (Max Error)**

Diferente da média, o erro máximo mostra o pior cenário. Em redes neurais, um único outlier causado por um arredondamento ruim em uma camada pode se propagar e destruir a predição final. Então precisamos verificar se o `scale` está saturando demais os valores. Como não estou utilizando granularidade por canal, isso com certeza vai ser alto.

**Desvio de Ativação Média (Mean Bias Drift)**

Média de $(A_{pure} - A_{fp32})$. Se a média for muito diferente de zero, a quantização está introduzindo um viés sistemático.
Aqui vai ser possível visualizar o efeito do `trunc`, que geralmente desloca a média da ativação para baixo, enquanto o `round` tende a manter o erro centrado em zero.

### Estimativa de custo

TDP (Thermal Design Power) representa a capacidade máxima de dissipação térmica do componente. É o consumo máximo teórico. 
No cálculo assumo que, durante a inferência, o processador está operando em sua carga máxima de energia. Na prática, uma inferência rápida de matrizes pequenas não levaria o hardware a consumir 100% do TDP o tempo todo. O consumo real varie, mas ainda sim servecomo uma métrica teórica que aparentemente é o padrão da indústria para estimar o "pior cenário" de custo energético e necessidades de resfriamento.

Para uma estimativa real na AWS por exemplo, o consumo flutua conforme a carga, assim como o custo financeiro, mas utiliza taxas horárias fixas das instâncias específicas (c7i.xlarge e g4dn.xlarge).
Estou convertendo o tempo de execução em frações de hora e multiplicando pelo preço da instância.
Isso é útil para comparar a eficiência entre os modos (FP32 vs Quantizado), mas não reflete a conta real da AWS, que geralmente cobra por hora cheia ou segundos (com mínimos), além de custos de transferência de dados e armazenamento que não estão no script.
3. Terei que olhar "aquele lance de carbono"?
Depende do seu objetivo. Se você precisa de um relatório de Sustentabilidade (ESG) ou quer saber o impacto ambiental real, o cálculo de energia puro não é suficiente.

Devemos incluir questões como a refrigeração do data center, perdas na rede elétrica e fabricação do hardware? Localidade? Considerar a matriz energética da região? São métricas que a Amazon considera.

### CCFT

O AWS Customer Carbon Footprint Tool é uma ferramenta utilizada para entender e relatar a pegada de carbono total das nossas cargas de trabalho na AWS ao longo do tempo. Fornece uma visão agregada das emissões de CO2e (dióxido de carbono equivalente) associadas ao uso de serviços AWS, considerando fatores como o tipo de serviço, a região da AWS e a intensidade de carbono da matriz energética local.

No entanto, o CCFT não foi projetado para medição em tempo real, então ele não oferece uma API para consultar a pegada de carbono de uma execução de código específica e de curta duração, como uma única inferência de modelo. Os dados são agregados em níveis de conta e região, não permitindo associar emissões diretamente a blocos de código ou funções individuais. Embora seja possível exportar os dados de emissão de carbono para o Amazon S3 através do serviço AWS Data Exports, isso é mais adequado para análise histórica e relatórios de sustentabilidade, e não para instrumentar o código para medir o impacto de cada execução. Acredito que para o nosso caso, onde o objetivo é comparar o impacto energético e de carbono de diferentes modos de inferência (FP32 vs. quantizado) em tempo de execução, uma ferramenta que possa monitorar o consumo de energia localmente ou estimá-lo com base no hardware e na localização é mais apropriada.

Fui procurar essa tal ferramenta e achei o CodeCarbon`, uma biblioteca de código aberto que estima as emissões de carbono geradas pela execução de código. Ele faz isso monitorando o consumo de energia do hardware (CPU, GPU) e multiplicando-o pela intensidade de carbono da eletricidade na região onde o código está sendo executado.

Utiliza bibliotecas como pynvml (para GPUs NVIDIA) e psutil (para CPUs) para estimar o consumo de energia do sistema durante a execução do código. Tenta determinar a localização geográfica da execução para identificar a matriz energética local. Com base na localização, ele consulta um banco de dados de intensidade de carbono (gCO2e/kWh) para calcular as emissões de CO2e.

```txt
========================================================================================
ESCALAS
========================================================================================
scale_x    : dtype=float32,  value=0.015430472791194916
scale_w    : dtype=float32,  value=0.007787053007632494
scale_bias : dtype=float32,  value=0.00012015790707664564
scale_y    : dtype=float32,  value=0.012964674271643162
========================================================================================
ENTRADAS QUANTIZADAS - linha 0
========================================================================================
round
FP32          : dtype=float32, value=[ 0.49671414 -0.1382643   0.64768857  1.5230298  -0.23415338]
INT8 quant    : dtype=int8,  value=[ 32  -9  42  99 -15]
FP32 dequant  : dtype=float32, value=[ 0.49377513 -0.13887426  0.6480799   1.5276169  -0.23145708]
Comp. a 2     : ['00100000', '11110111', '00101010', '01100011', '11110001']
========================================================================================
trunc
FP32          : dtype=float32, value=[ 0.49671414 -0.1382643   0.64768857  1.5230298  -0.23415338]
INT8 quant    : dtype=int8,  value=[ 32  -8  41  98 -15]
FP32 dequant  : dtype=float32, value=[ 0.49377513 -0.12344378  0.63264936  1.5121863  -0.23145708]
Comp. a 2     : ['00100000', '11111000', '00101001', '01100010', '11110001']
========================================================================================
INFERENCIA FP32 - linha 0
========================================================================================
Z_fp32 (entrada ativacao) : [ 0.5592872  -0.43355328 -2.142712   -0.36288917  0.31692466]
A_fp32 (saída  ativacao)  : [0.5592872  0.         0.         0.         0.31692466]
========================================================================================
INFERENCIA HIBRIDA - linha 0
========================================================================================
[round] - entradas X:
round
FP32          : dtype=float32, value=[ 0.49671414 -0.1382643   0.64768857  1.5230298  -0.23415338]
INT8 quant    : dtype=int8,  value=[ 32  -9  42  99 -15]
FP32 dequant  : dtype=float32, value=[ 0.49377513 -0.13887426  0.6480799   1.5276169  -0.23145708]
Comp. a 2     : ['00100000', '11110111', '00101010', '01100011', '11110001']

[round]
Z - entrada ativacao:
FP32         : [ 0.564347   -0.43586928 -2.142623   -0.36818796  0.311674  ]
INT8 quant   : [ 22 -17 -82 -14  12]
FP32 dequant : [ 0.5731531  -0.44289103 -2.136298   -0.3647338   0.31262895]
Comp. a 2    : ['00010110', '11101111', '10101110', '11110010', '00001100']
A - saida  ativacao:
FP32         : [0.564347 0.       0.       0.       0.311674]
UINT8 quant  : [43  0  0  0 24]
FP32 dequant : [0.5579303  0.         0.         0.         0.31140298]
========================================================================================
[trunc] - entradas X:
trunc
FP32          : dtype=float32, value=[ 0.49671414 -0.1382643   0.64768857  1.5230298  -0.23415338]
INT8 quant    : dtype=int8,  value=[ 32  -8  41  98 -15]
FP32 dequant  : dtype=float32, value=[ 0.49377513 -0.12344378  0.63264936  1.5121863  -0.23145708]
Comp. a 2     : ['00100000', '11111000', '00101001', '01100010', '11110001']

[trunc]
Z - entrada ativacao:
FP32         : [ 0.54287225 -0.43276525 -2.1150506  -0.3441162   0.3129307 ]
INT8 quant   : [ 21 -16 -81 -13  12]
FP32 dequant : [ 0.5419802  -0.4129373  -2.090495   -0.33551157  0.309703  ]
Comp. a 2    : ['00010101', '11110000', '10101111', '11110011', '00001100']
A - saida  ativacao:
FP32         : [0.54287225 0.         0.         0.         0.3129307 ]
UINT8 quant  : [42  0  0  0 24]
FP32 dequant : [0.5398548  0.         0.         0.         0.30848846]
========================================================================================
INFERENCIA QUANTIZADA PURA - linha 0
========================================================================================
Pesos W quantizados:
W_q_round  INT8 : [ -92   78 -109  125   70]
W_q_round Comp2 : ['10100100', '01001110', '10010011', '01111101', '01000110']
W_q_trunc  INT8 : [ -92   77 -109  125   69]
W_q_trunc Comp2 : ['10100100', '01001101', '10010011', '01111101', '01000101']
b_q (INT32)     : [1775 2170  510 2255  -52]
========================================================================================
[round]
Z_int32_round (INT32)    : [  4747  -3551 -17792  -3126   2578]
Y_q_round     (UINT8)    : [44  0  0  0 24]
A_pure_round  (FP32 dq)  : [0.57044566 0.         0.         0.         0.3111522 ]
Métricas [pure-round vs FP32]
MSE:        0.00006161
Max Error:  0.02417076
SQNR (dB):  42.40
Bias Drift: 0.00053170
========================================================================================
[trunc]
Z_int32_trunc (INT32)    : [  4444  -3558 -17563  -2786   2648]
Y_q_trunc     (UINT8)    : [41  0  0  0 25]
A_pure_trunc  (FP32 dq)  : [0.53155166 0.         0.         0.         0.32411686]
Métricas [pure-trunc vs FP32]
MSE:        0.00018501
Max Error:  0.03889394
SQNR (dB):  37.62
Bias Drift: -0.00620993
========================================================================================
ESTIMATIVA DE CUSTO ENERGETICO E DE CARBONO
========================================================================================
Modo                    t (us)      CPU (nJ)      GPU (nJ)    AWS-CPU (USD)    AWS-GPU (USD)   Emissões (kgCO2e)
  --------------------------------------------------------------------------------------------------------------
  FP32                   74.5000  9312500.026226  22350000.062943       4.1720e-09       1.0885e-08        9.768335e-07
  Hibrida Round          68.6000  8574999.981192  20579999.954862       3.8416e-09       1.0023e-08        1.341761e-06
  Hibrida Trunc          44.5000  5562500.007272  13350000.017454       2.4920e-09       6.5019e-09        9.928460e-07
  Pura Round            103.1000  12887500.020042  30930000.048102       5.7736e-09       1.5064e-08        9.835711e-07
  Pura Trunc             87.5000  10937500.007913  26250000.018990       4.9000e-09       1.2785e-08        1.327555e-06
```

## Referências interessantes

- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/inferentia2.html
- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v2.html#neuroncores-v2-arch