# Teoria da Quantização Linear Assimétrica

## RESUME

Vamos começar recapitulando o que falei na semana passada (e também definindo um roteiro melhor para este aprendizado). Pelo que entendi, o objetivo central aqui é um só: encolher modelos (shrink models) mantendo sua eficácia.

## MODEL COMPRESSION TECHNIQUES

Existem três pilares principais para tornar um modelo mais eficiente:
- Quantização: Representar os parâmetros do modelo em uma precisão menor (ex: de float32 para int8).
- Destilação de Conhecimento (Knowledge Distillation): Treinar um modelo "aluno" menor usando as saídas de um modelo "professor" maior.
- Poda (Pruning): Remover conexões (pesos) desnecessárias dentro do modelo para torná-lo esparso.

Como o foco é matemática em ponto fixo, vou "abrir o capô" da quantização e tentar construir algumas ferramentas próprias para não ficarmos apenas na teoria.

## QUANTIZATION CONCEPT

Quantizar é o processo de mapear um conjunto grande de valores para um conjunto menor. No contexto de redes neurais, focamos na **quantização linear**, que utiliza um mapeamento linear para converter valores de alta precisão (como `float32`) para baixa precisão (como `int8`).

## QUANTIZATION TYPES

Uma observação importante que não comentei da última vez é o fato de estar estudando a quantização linear. Então pera ai um instante, existe a quantização não linear? Existe, embora seja mais complexa de implementar de forma eficiente nos hardwares atuais.

Para entender a diferença, imagine que a quantização linear funciona como uma régua comum, onde as marcações de centímetros são todas iguais. Já a não linear funciona como uma régua "elástica", onde algumas partes são mais detalhadas que outras.

### Quantização Linear: 
Na quantização linear, os níveis de quantização são distribuídos de forma uniforme ao longo de todo o intervalo dos dados.
- **Vantagem**:
É matematicamente simples, fácil de implementar e extremamente eficiente para o hardware, pois se reduz basicamente a operações de escala, arredondamento e clipping, todas suportadas diretamente por instruções vetoriais e unidades especializadas.
- **Desvantagem**:
Quando a maior parte dos valores está concentrada próxima de zero, que é um cenário comum em pesos e ativações de redes neurais, a quantização linear acaba desperdiçando níveis de precisão em regiões extremas do intervalo que são pouco utilizadas, aumentando o erro de quantização nos valores mais relevantes.
    
### Quantização Não Linear:
Na quantização não linear, os intervalos entre os níveis de quantização não são uniformes. A ideia central é alocar mais níveis (ou bits efetivos) para valores que aparecem com maior frequência e menos níveis para valores raros, reduzindo o erro médio de quantização. Alguns exemplos comuns incluem:

- **Quantização Logarítmica**:
Em vez de mapear os valores de forma linear, aplica-se uma transformação logarítmica antes da quantização. Esse método é amplamente utilizado em processamento de áudio, como nas leis μ-law e A-law, pois o ouvido humano é muito mais sensível a variações em sons de baixa intensidade do que em sons altos.

- **Quantização Baseada em Agrupamento (Clustering / K-Means):**
Os pesos do modelo são agrupados em $k$ clusters, e cada valor é representado pelo centroide do seu grupo. Esse método pode ser extremamente preciso, porém requer uma tabela de consulta (lookup table) para mapear índices aos valores reais, o que tende a aumentar a latência e o custo computacional.

- **Normal Float (NF4):**
Utilizada no algoritmo QLoRA para modelos de linguagem de grande porte. A NF4 assume que os pesos seguem aproximadamente uma distribuição normal (Gaussiana) e posiciona os níveis de quantização de modo que cada nível tenha a mesma probabilidade de ocorrência, maximizando a eficiência estatística da representação.

Na linguagem de processamento de sinais, coisa que vocês dois são mais do que especialistas, fica mais legal. Com a quantização linear, cada incremento no valor amostrado corresponde a um incremento analógico de tamanho fixo. Por exemplo, um conversor AD ou DA de 8 bits com uma faixa analógica de 0 a 1 V tem $1/256 = 3,9 mV$ por bit, independentemente da amplitude real do sinal.
Com a quantização não linear, normalmente se utiliza algum tipo de codificação logarítmica (Lei µ/Lei A), de modo que o incremento para valores de amostra pequenos seja muito menor do que o incremento para valores de amostra grandes. Idealmente, o tamanho do passo deve ser aproximadamente proporcional ao tamanho da amostra. Isso se traduz em uma relação sinal/ruído fixa (devido ao ruído de quantização), independentemente da amplitude do sinal. Outra maneira de ver isso é que você pode usar menos bits para obter uma determinada relação sinal/ruído na faixa de amplitude do sinal de interesse.

## LINEAR INTERPOLATION APPROACH

### Quantização Linear

Dado um sinal contínuo $(x \in [x_{\min}, x_{\max}])$ e um número de bits $(b)$:

#### Número de níveis

$$L = 2^b$$

#### Passo de quantização

$$\Delta = \frac{x_{\max} - x_{\min}}{L - 1}$$

#### Quantização

$$Q_{\text{linear}}(x) = \Delta \cdot \operatorname{round}!\left(\frac{x - x_{\min}}{\Delta}\right) + x_{\min}$$

No código: $x_{\min} = -1,\quad x_{\max} = 1$

Onde:
- *L*: Número total de níveis de quantização disponíveis (ex: 256 para 8 bits).
- *b*: Número de bits utilizados na representação (profundidade de bits).
- $\Delta$: Tamanho do passo de quantização (ou *step size*); é a distância entre dois níveis consecutivos.
- *$x_{max}$* e *$x_min$*: Valores máximo e mínimo do sinal original (fundo de escala).
- *x*: O valor de entrada original (valor real/contínuo).
- *$Q_{linear}(x)$**: O valor final já quantizado e reconstruído na escala original.
- *round*: Função de arredondamento para o inteiro mais próximo.

### Quantização Não Linear (μ-law)

A quantização μ-law ocorre em **três etapas**: compressão, quantização linear e expansão.

#### Compressão μ-law

Para um parâmetro ($\mu > 0$):

$$x_c = \operatorname{sign}(x),\frac{\ln!\left(1 + \mu |x|\right)}{\ln(1 + \mu)}$$

Isso comprime valores pequenos com maior resolução.

#### Quantização linear no domínio comprimido

Com $(b)$ bits:

$$L = 2^b$$

$$\Delta_c = \frac{2}{L - 1}$$

$$Q_c(x_c) = \Delta_c \cdot \operatorname{round}!\left(\frac{x_c + 1}{\Delta_c}\right) - 1$$

#### Expansão (μ-law inverso)

$$Q_{\mu\text{-law}}(x) =
\operatorname{sign}(Q_c),
\frac{(1 + \mu)^{|Q_c|} - 1}{\mu}$$

Esta abordagem aplica uma deformação no sinal antes de quantizar para favorecer valores menores.

- *$x_c$*: Valor do sinal após a fase de **Compressão** (logarítmica).
- *$\mu$*: Fator de compressão (parâmetro que define o quão "curvada" será a escala; valores comuns são 100 ou 255).
- *sign(x)*: Função sinal (retorna  se o valor for positivo e  se for negativo).
- *$Q_c$*: Valor quantizado no domínio comprimido.
- *$\Delta_c$*: Passo de quantização específico para o sinal comprimido.
- *$Q_\mu$*: Valor final após a **Expansão** (recuperação do sinal original corrigido).

## DSP APPROACH

Variáveis específicas da visão de processamento de sinais clássico.

* *$X_m$*: Representa a amplitude máxima do sinal (geralmente assume-se que o sinal vai de $-X_m$ até $+X_m$ ).
* *$\lfloor \dots \rfloor$*: Função *floor* (piso), que arredonda para baixo (usada aqui junto com  para simular o arredondamento).
* *e(n)*: Erro (ou ruído) de quantização; a diferença entre o sinal real e o quantizado.
* *x[n]*: Amostra do sinal original no tempo discreto

## ML APPROACH

1. **Pesos:** Os parâmetros fixos do modelo.
2. **Ativações:** Os valores que fluem através das camadas durante a inferência.

## LINEAR QUANTIZATION EXAMPLE

A relação fundamental da quantização linear é expressa pela fórmula:

$$r = S \cdot (Q - Z)$$

Onde:
- *r*: Valor real original (ex: `float32`).
- *Q*: Valor quantizado (ex: `int8`).
- *S*: Fator de escala (mesmo tipo de dado de *r*).
- *Z*: O valor no domínio quantizado que corresponde ao zero real (mesmo tipo de *Q*).

### Como obter o valor quantizado?

Para transformar um valor real em um inteiro de 8 bits, isolamos o *Q* na equação acima:

1. Dividimos o valor real pelo escala
2. Somamos o ponto zero
3. Arredondamos para o inteiro mais próximo
4. Garantimos que o valor esteja dentro do intervalo do `int8` (entre -128 e 127).

## TRADE-OFFS

* **Vantagens:** Modelos menores, menor consumo de memória e ganhos de velocidade em operações de matriz (GEMM).
* **Desafios:** Perda de precisão e necessidade de encontrar parâmetros ideais para o mapeamento.

## LINEAR QUANTIZATION MODES

Temos dois modos de quantização linear, o simétrico e o assimétrico


---

##

Quantização por tensor: usa um único fator de escala (e zero-point) para todo o tensor. É simples, rápida e eficiente em hardware, mas pode perder precisão quando os valores variam muito dentro do tensor.

Quantização por canal: aplica um fator de escala diferente para cada canal (ex.: canais de saída em camadas convolucionais). Mantém melhor a precisão do modelo, especialmente em pesos, ao custo de maior complexidade e uso de memória.

Quantização por grupo: divide o tensor em pequenos grupos e aplica uma escala por grupo. Fica no meio-termo: mais precisa que a quantização por tensor e mais eficiente que a por canal, equilibrando desempenho e qualidade.

Vou implementar e comparar diferentes esquemas de quantização:
- Per-tensor: Um único fator de escala para o tensor inteiro.
- Per-channel: Escalas diferentes para cada canal (comum em pesos de convolução).
- Per-group: Divisão do tensor em grupos menores para maior precisão.
Agora vamos criar um quantizador de 8 bits universal. Como esse esquema é agnóstico à modalidade, ele funcionará em qualquer modelo que utilize camadas lineares, seja ele de:
- Visão Computacional (Imagens)
- NLP (Texto/Linguagem)
Áudio ou Multimodal

3. Desafios da Quantização Extrema (2-bit e 4-bit)
À medida que tentamos reduzir os modelos para 4 bits ou até 2 bits, surgem novos desafios técnicos. O PyTorch, nativamente, não suporta tipos de dados tão pequenos.
Weight Packing (Empacotamento de Pesos): Aprenderemos a técnica de "empacotar" vários pesos de baixa precisão dentro de um único tensor de maior precisão (como um int8) para economizar memória.
Algoritmos de Packing/Unpacking: Vamos implementar a lógica para guardar e recuperar esses valores comprimidos.
4. Estado da Arte e LLMs
Finalizaremos o curso discutindo os desafios específicos de quantizar modelos imensos, como os LLMs (Large Language Models), e revisaremos os métodos mais modernos utilizados pela indústria atualmente.

## Experimento

### Baseline

ResNet50 é uma rede neural convolucional (CNN) de 50 camadas, introduzida pela Microsoft Research em 2015, amplamente usada para visão computacional. Ela utiliza conexões residuais ("skip connections") para resolver o problema do vanishing gradient em redes profundas, permitindo treinamento eficiente. Famosa pela alta precisão em classificação de imagens, detecção de objetos e segmentação, sendo frequentemente pré-treinada no conjunto de dados ImageNet. 

Características Principais da ResNet50: 
- Estrutura: Composta por 50 camadas, incluindo camadas de convolução, blocos residuais e uma camada totalmente conectada no final.
- Conexões Residuais: Permitem que a saída de um bloco anterior seja adicionada a um bloco posterior, facilitando o fluxo de gradientes e permitindo redes mais profundas.
- Entrada de Imagem: Padrão de \(224\times 224\) pixels, com 3 canais de cor (RGB).
- Aplicações: Utilizada em classificação de imagens, detecção de objetos, segmentação de instâncias, e também em diagnósticos médicos (ex: mamografias) e automação industrial.
- Pré-treinamento: Disponível em bibliotecas como TensorFlow/Keras e PyTorch com pesos treinados no ImageNet, permitindo transfer learning para novas tarefas. 

A arquitetura se destaca por permitir o treinamento de redes muito profundas sem perda significativa de desempenho, sendo um pilar fundamental no aprendizado profundo moderno.

### Concorrente

a

### Conjuntos de dados

Neste experimento, vamos analisar diferentes conjuntos de dados, abrangendo tanto problemas de classificação tabular quanto de Visão Computacional, permitindo avaliar o comportamento dos métodos estudados em cenários de distintas complexidades e dimensionalidades.

#### Classificação Tabular

**Iris Dataset**
- 150 amostras
- 4 atributos contínuos (comprimento e largura da sépala e da pétala)
- 3 classes
- Conjunto de dados clássico, de baixa dimensionalidade, amplamente utilizado para a avaliação e comparação de algoritmos de classificação

**Breast Cancer Wisconsin (Diagnostic)**
- 569 amostras
- 30 atributos numéricos contínuos
- 2 classes (benigno e maligno)
- Dataset voltado para classificação binária, frequentemente empregado em estudos de aprendizado supervisionado e avaliação de desempenho de modelos

**Wine Dataset**
- 178 amostras
- 13 atributos contínuos relacionados à composição química do vinho
- 3 classes
- Conjunto de dados multiclasse e de baixa dimensionalidade, adequado para analisar o impacto de técnicas como quantização em dados tabulares

### Classificação em visão computacional

Problemas de classificação de imagens, que apresentam maior dimensionalidade e complexidade estrutural.

**MNIST**
- 70.000 imagens no total (60.000 para treinamento e 10.000 para teste)
- Imagens em escala de cinza
- Resolução de 28 × 28 pixels
- 10 classes correspondentes aos dígitos de 0 a 9
- Benchmark amplamente utilizado para avaliação de algoritmos de aprendizado de máquina e redes neurais em tarefas de classificação multiclasse

**Fashion-MNIST**
- 70.000 imagens em escala de cinza (60.000 para treinamento e 10.000 para teste)
- Resolução de 28 × 28 pixels
- 10 classes representando diferentes categorias de vestuário
- Alternativa mais desafiadora ao MNIST, projetada para avaliação de modelos em cenários de maior complexidade visual

**EMNIST (Extended MNIST)**
- Extensão do MNIST contendo letras e dígitos
- Imagens em escala de cinza com resolução de 28 × 28 pixels
- Conjunto de dados multiclasse com maior número de classes
- Utilizado para avaliar a capacidade de generalização de modelos em problemas com maior diversidade de padrões

**CIFAR-10**
- 60.000 imagens coloridas (50.000 para treinamento e 10.000 para teste)
- Resolução de 32 × 32 pixels
- 10 classes de objetos do cotidiano
- Dataset amplamente adotado como benchmark em Visão Computacional, apresentando maior complexidade devido à variação de cores, texturas e formas.

## Extra

### Diferença no hardware

A diferença entre usar **FP32** e **INT8** no nível de memória e registradores é, literalmente, uma questão de **densidade e "vazão"**. Imagine que a memória é um armazém e os registradores são a mesa de trabalho do processador.

#### 1. Na Memória

Cada número que sua MLP precisa processar (pesos e ativações) ocupa um espaço físico nos chips de RAM.

- **FP32 (32 bits/4 bytes):** É como guardar cada item em uma caixa grande. Se você tem uma MLP com 1 milhão de parâmetros, você precisa de **4 MB** de memória apenas para os pesos.
- **INT8 (8 bits/1 byte):** É como compactar esse item para caber em uma caixa 4x menor. O mesmo modelo agora ocupa apenas **1 MB**.

O processador consegue buscar 4 números `INT8` no mesmo tempo que gastaria para buscar apenas 1 número `FP32`. Isso reduz o "engarrafamento" de dados (banda de memória), que podemos considerar o maior gargalo em aprendizado profundo.

#### 2. Nos Registradores

Os registradores são pequenos espaços dentro do processador onde as contas acontecem de fato. Eles têm um tamanho fixo (geralmente 256 bits ou 512 bits em CPUs modernas com tecnologia **SIMD**).

* **Com FP32:** Em um registrador de 256 bits, você só consegue "sentar à mesa" **8 números** por vez (). O processador faz 8 multiplicações simultâneas.
* **Com INT8:** No mesmo registrador de 256 bits, você consegue colocar **32 números** ().
* **O Efeito Prático:** O hardware faz o **quádruplo de trabalho** no mesmo ciclo de clock. É como se você tivesse 4 operários trabalhando simultaneamente na mesma mesa onde antes só cabia um.

<details><summary><b>SIMD</b></summary>
SIMD (Single Instruction, Multiple Data) é uma tecnologia que permite ao processador aplicar uma única instrução (como uma soma ou multiplicação) a vários dados simultaneamente.

Em vez de processar um número por vez, o registrador funciona como um "vetor", realizando a operação em lote. Por isso, ao mudar para INT8, o ganho é imediato: como os dados são menores, cabem muito mais números dentro do mesmo registrador SIMD, permitindo que o hardware processe até 4x mais informações no mesmo ciclo de clock.
</details>

#### 3. O Acumulador

Embora a *entrada* seja `INT8`, o hardware geralmente usa um registrador maior (como `INT32`) para somar os resultados (**Acumulador**).

- **Por que?** Se você multiplicar dois números de 8 bits, o resultado pode ter até 16 bits. Se você somar vários desses resultados (o que uma MLP faz o tempo todo), o valor "transborda" rapidamente os 8 bits originais.
- Portanto, o registrador de trabalho "engorda" temporariamente durante a soma para garantir que você não perca ainda mais precisão por *overflow*.

Eu fiz um código que explica de forma visual como o tamanho dos dados impacta diretamente o consumo de VRAM e a eficiência de hardware para LLMs.

Vamos tentar estimar quanta memória (em GB) é necessária para armazenar apenas os parâmetros de um modelo de linguagem, variando:
- Tamanho do modelo (em bilhões de parâmetros)
- Tipo numérico usado para armazenar cada parâmetro (FP32, FP16, INT8, etc.)

Não estou considerando:
- Memória de ativação
- Gradientes
- Otimizador
- KV cache
- Overhead do framework

Ou seja, é o **mínimo teórico** para armazenar os pesos.

Antes de mais nada, acho importante sair do “o que o código faz” e ir para por que ele é feito assim, do ponto de vista matemático, computacional e físico.

#### Descrição matemática

#### O que é um modelo de ML fisicamente?

No nível mais baixo possível, um modelo é:

**Uma sequência de números armazenados na memória**

Nada mais.

Esses números são:
- pesos
- vieses
- embeddings
- matrizes de atenção
- etc.

Ou seja:
- Modelo = lista de parâmetros
- Parâmetro = número
- Número = bits na memória

Se você não sabe quantos bits cada número ocupa, você não sabe quanta memória o modelo precisa. Esse é o ponto central.

#### O único problema que o algoritmo resolve

O algoritmo responde uma única pergunta:

**“Quantos bytes eu preciso para armazenar N números, se cada número ocupa B bytes?”**

Memória total = número de parâmetros × tamanho de cada parâmetro

Essa equação não é uma escolha, é simplesmente uma identidade física.

#### Código

Para o tamanho dos modelos, considere:
| Valor | Significado            |
| ----- | ---------------------- |
| 1     | 1 bilhão de parâmetros |
| 7     | 7 bilhões              |
| 13    | 13 bilhões             |
| 70    | 70 bilhões             |

Como imagino que já saibam, em LLMs, o número de parâmetros é o principal fator que determina:
- Capacidade do modelo
- Custo computacional
- Consumo de memória

Para os tipos de dados, considere:

| Tipo | Bits | Bytes | Observação                   |
| ---- | ---- | ----- | ---------------------------- |
| FP32 | 32   | 4     | Alta precisão, padrão antigo |
| FP16 | 16   | 2     | Muito usado em GPUs modernas |
| BF16 | 16   | 2     | Melhor estabilidade numérica |
| INT8 | 8    | 1     | Quantização, pouca perda     |
| INT4 | 4    | 0.5   | Quantização agressiva        |

Aqui você define quanto espaço cada parâmetro ocupa. Quanto menor o tipo, menos memória, mas maior o risco de perda de qualidade.

**Cálculo de memória total em bytes**

1. A função recebe o tamanho do modelo em bilhões e tamanho de cada parâmetro em bytes
2. Converte para número absoluto de parâmetros multiplicando por 1 bilhão.
3. Multiplica isso pela quantidade de bytes por parâmetro
4. Converte para Gigabytes 
    - Usa GiB reais -> 1 GiB = $1024^3$ bytes ≈ 1.07 GB
    - Isso é importante para estimar uso real de VRAM/RAM.

```python
import pandas as pd

model_sizes = [1, 7, 13, 70]

data_types = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "INT8": 1,
    "INT4": 0.5
}

def calculate_model_memory(num_params_billions, bytes_per_param):
    num_params = num_params_billions * 1e9
    bytes_total = num_params * bytes_per_param
    gb = bytes_total / (1024 ** 3)
    return gb

results = []

for num_params_b in model_sizes:
    row = {"Model Size": f"{num_params_b}B"}
    for dtype, bytes_size in data_types.items():
        memory_gb = calculate_model_memory(num_params_b, bytes_size)
        row[dtype] = f"{memory_gb:.2f} GB"
    results.append(row)
    
df = pd.DataFrame(results)
display(df)
```

#### Observações:

1. FP32 é inviável para LLMs grandes
- 70B FP32 → ~260 GB
- Só possível em servidores multi-GPU

2. FP16/BF16 é o padrão atual
- Bom equilíbrio entre precisão e custo
- Treinamento e inferência moderna usam isso

3. Quantização muda tudo
- INT8 e INT4 tornam modelos grandes viáveis em GPUs menores
- Exemplo: 13B INT4 → ~6 GB → cabe em uma GPU de 8 GB

Isso que mostrei é só o começo. Na prática, você precisa multiplicar isso por:

- ×2 a ×4 (inferência)
- ×6 a ×8 (treinamento)

E por ai vai...

- Estender para treinamento vs inferência
- Incluir KV cache
- Calcular quantas GPUs são necessárias
- Simular LoRA/QLoRA

### Biblioteca torchao

#### Informações de Hardware

```python
import torch
print(torchao.__version__)
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())

# Result:
# 0.15.0
# NVIDIA GeForce RTX 3060
# (8, 6)
```

A RTX 3060 suporta Tensor Cores. Ela é Ampere (SM 8.6), a mesma geração das RTX 30.

O que os Tensor Cores da RTX 3060 suportam
- FP16 ✅
- BF16 ✅
- TF32 (automático no PyTorch para GEMM/Conv) ✅
- INT8 ✅
- INT4 ✅ (via kernels específicos, como os usados pelo torchao/TinyGEMM/Marlin)

#### Instalação

Para fazer esse experimento aqui, eu utilizei o Docker para buildar uma imagem Linux.

Vale apena falar sobre "Dependency Hell". Para fazer isso funcionar direito foi um parto.

O problema não é o hardware, mas o ritmo de desenvolvimento. O torchao é um projeto de "pesquisa ativa" do time do PyTorch. Ele serve para testar as funcionalidades mais novas de quantização que ainda nem entraram na versão principal do PyTorch.

- TorchAO 0.15.0: Foi construído usando APIs que só existem no código-fonte do PyTorch 2.6 (que ainda é experimental ou "Nightly").
- PyTorch 2.5.1: É a versão estável atual. Ela ainda não conhece os novos tipos de dados (como o int1) que o TorchAO 0.15.0 exige.

É como tentar instalar um jogo de 2026 em um Windows de 2024, vai dar merda. O SO simplesmente não tem os arquivos de sistema necessários para rodar o software novo.

Para a RTX 3060, tenho suporte a bfloat16 e int4. Estou utilizando o CUDA 13.0 (recente), então me restaram o número incrível de duas combinações possíveis para evitar erros:

Obs.: Me sinto combinando plantas no Resident Evil...

Opção A:
- PyTorch: 2.5.1
- CUDA: 12.1 ou 12.4 (O PyTorch ainda não tem pacotes oficiais compilados especificamente para o CUDA 13, mas o 12.4 roda perfeitamente no seu driver 13).
- TorchAO: 0.6.1

Opção B:
- PyTorch: 2.6.0.dev (Nightly)
- TorchAO: 0.15.0
Obs.: Eu acho que a opção B não me cheira bem... mas se você realmente precisar de uma função que só tem no 0.15.0, vai ter atualizar o PyTorch para a versão de testes e se fud..

#### Entendendo a biblioteca

A função `quantize_` é o "coração" da biblioteca torchao. Diferente das abordagens antigas do PyTorch, ela foi desenhada para ser extremamente simples: você define o que quer e ela aplica isso diretamente na estrutura do seu modelo.

O <i>underline</i> no nome da função (`quantize_`) segue a convenção do PyTorch para operações `in-place`. Isso significa que ela altera o objeto model que você passa como argumento. Você não precisa fazer `model = quantize_(model)`, o modelo original já sairá transformado.

### Detalhamento dos parâmetros

- model (O modelo de entrada): É o seu nn.Module (uma ResNet, um Transformer, etc.). A função percorrerá todas as "folhas" (sub-módulos) desse modelo em busca de camadas que possam ser quantizadas (geralmente camadas Linear ou Conv2d).

- config (Configuração do Workflow): Este é o objeto que define a estratégia. É aqui que você diz à biblioteca: "Quero que os pesos virem int8, mas mantenha as ativações em float16".

- filter_fn (O Filtro de Precisão): Este é o parâmetro que permite ao usuário escolher quais partes do modelo quer quantizar.
    - Exemplo: Você pode querer quantizar todas as camadas lineares, exceto a última camada (o "head" de classificação), para manter a precisão final alta.
    - A função recebe o módulo e o nome completo dele (FQN - Fully Qualified Name) e retorna True para quantizar ou False para ignorar.

- device (Dispositivo): Se você estiver quantizando um modelo grande, fazer isso na CPU pode ser lento. Se você passar device="cuda", o torchao moverá o módulo para a GPU, aplicará a transformação (que muitas vezes envolve cálculos para encontrar escalas de quantização) e entregará o modelo pronto lá.

A parte do `config` é realmente onde a "mágica" acontece. No `torchao`, o objeto de configuração não é apenas um dicionário de opções; ele é um **blueprint (plano de execução)** que dita como os tensores de peso e ativação serão transformados e quais kernels (algoritmos matemáticos) serão usados para as operações.

Quando você passa um `config` para a função `quantize_`, o `torchao` realiza três etapas principais baseadas nele:

1. **Mapeamento:** Ele identifica quais tipos de camadas (ex: `nn.Linear`) são compatíveis com aquela configuração.
2. **Transformação de Tensores:** Ele converte os pesos de `float32/16` para o formato alvo (`int8`, `int4`, etc.) e calcula as escalas de quantização.
3. **Substituição de Operadores:** Ele substitui a multiplicação de matrizes padrão por uma versão otimizada que entende os novos tipos de dados.

Por coincidência (ou não) existem três configurações que você que representam os principais caminhos de otimização da biblioteca hoje:

#### 1. `Int8WeightOnlyConfig` (Apenas Pesos)

Esta é a forma mais simples de quantização.

* **O que faz:** Transforma apenas os pesos fixos do modelo em `int8`. As ativações (os dados que fluem pelo modelo durante a inferência) continuam em `float16` ou `bfloat16`.
* **Vantagem:** Reduz o uso de memória do modelo em quase 50% (se comparado a `float16`) sem precisar de dados de calibração complexos.
* **Performance:** Excelente para modelos onde o gargalo é a leitura de memória (memory-bound), como LLMs em batch size 1.

#### 2. `Int4WeightOnlyConfig` (Compressão Agressiva)

* **O que faz:** Comprime os pesos para apenas 4 bits. Isso permite que um modelo que ocupava 16GB em `float16` caiba em cerca de 4GB a 5GB.
* **Destaque:** Utiliza o kernel **tinygemm**, que é altamente otimizado para descompactar esses 4 bits "on-the-fly" durante o cálculo, mantendo a velocidade alta. É ideal para rodar modelos grandes em GPUs com pouca VRAM (como GPUs de consumo ou dispositivos edge).

#### 3. `Int8DynamicActivationInt8WeightConfig` (Quantização Total)

Esta é a configuração mais avançada e voltada para máxima performance de processamento.

* **Pesos:** São armazenados em `int8`.
* **Ativações:** São quantizadas para `int8` **dinamicamente** (em tempo de execução, enquanto os dados passam pela camada).
* **Vantagem:** Permite usar as unidades de hardware especializadas da GPU (Tensor Cores) para multiplicar inteiros, o que é muito mais rápido do que multiplicar floats.
* **Uso de `torch.compile`:** Esta config brilha quando usada com o compilador do PyTorch, pois ele funde a quantização das ativações com a operação matemática, eliminando o overhead.

Embora existam essas classes prontas, o `config` permite definir o **layout** dos dados. Por exemplo, você pode definir se a quantização é "per-channel" (uma escala para cada neurônio) ou "per-tensor" (uma escala para a camada toda).

> **Dica Importante:** O `torchao` foi feito para trabalhar em conjunto com o `torch.compile()`. Depois de usar o `quantize_(model, config)`, sempre execute `model = torch.compile(model)` para que o PyTorch possa gerar o código de máquina otimizado para os kernels que o `config` selecionou.

#### Conclusão prática

Era para ser óbvio desde o início, mas não foi... Nessa biblioteca não iremos conseguir ver quantização em CPU. Eu pensei: "Estou só quantizando, não inferindo…"

Só que nessa biblioteca, quantização ≠ só converter números.

O fluxo real é:
````txt
FP32 weight
 → quantização afim
 → geração de int4/int8 lógico
 → PACKING físico (2 valores por byte)
 → layout específico de TensorCore
```

Esse kernel:
- Assume GPU
- Assume Tensor Cores
- Assume layout CUDA

CPU nunca foi um alvo suportado. Int4WeightOnlyConfig NÃO é uma quantização genérica. É uma quantização orientada a kernel CUDA específico.

- Não serve para estudo interno
- Não serve para CPU
- Não serve para “ver o peso”