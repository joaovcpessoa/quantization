# Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

## 1. Introduction

Comenta sobre o aumento e popularidade modelos de Aprendizado Profundo e da necessidade de criar modelos cada vez mais eficientes. Eles propõe um esquema de quantização que permite que a inferência seja realizada usando aritmética somente de inteiros, o que pode ser implementado de forma mais eficiente do que a inferência de ponto flutuante em hardware comum disponível que utiliza apenas inteiros. Também trazem um procedimento de treinamento para preservar a precisão do modelo de ponta a ponta após a quantização. Como resultado, o esquema de quantização proposto diz melhorar a relação entre precisão e latência no dispositivo. As melhorias são significativas mesmo em MobileNets, uma família de modelos conhecida por sua eficiência em tempo de execução, e são demonstradas na classificação do ImageNet e na detecção de COCO em CPUs populares.

Interessante é que eles comentam que não enxergam como necessário trazer eficiência para qualquer modelo, pois consideram que alguns são por natureza superparametrizados, então eles focam em modelos que já carregam eficiência em sua arquitetura, como no caso das MobileNets.

Também fazem uma crítica de algumas abordagens de quantização não fornecem melhorias de eficiência verificáveis ​​em hardware real. Como quantização apenas os pesos, que focam principalmente com o armazenamento no dispositivo e menos com a eficiência computacional, com exceção das redes binárias, ternárias e de deslocamento de bits, que empregam pesos que são 0 ou potências de 2, o que permite que a multiplicação seja implementada por deslocamentos de bits. No entanto, embora possam ser eficientes em hardware personalizado, oferecem pouco benefício em hardware existente com instruções de multiplicação e adição que, quando usadas corretamente, não são mais caras do que apenas adições. Além disso, as multiplicações só são caras se os operandos forem largos, e a necessidade de evitar multiplicações diminui com a profundidade de bits, uma vez que tanto os pesos quanto as ativações são quantizados. Há também abordagens que quantizam tanto os pesos quanto as ativações em representações de 1 bit. Com essas abordagens, tanto multiplicações quanto adições podem ser implementadas por operações eficientes de deslocamento e contagem de bits, que são demonstradas em kernels de GPU personalizados. No entanto, a quantização de 1 bit geralmente leva a uma degradação substancial do desempenho e pode ser excessivamente rigorosa na representação do modelo.

Sobre as figuras.

Figura 1.1: a) Inferência somente com aritmética de inteiros de uma camada de convolução. A entrada e a saída são representadas como inteiros de 8 bits. A convolução envolve operandos inteiros de 8 bits e um acumulado inteiro de 32 bits. A adição do viés envolve apenas inteiros de 32 bits (seção 2.4). A não linearidade ReLU6 envolve apenas aritmética de inteiros de 8 bits. 

b) Treinamento com quantização simulada da camada de convolução. Todas as variáveis ​​e cálculos são realizados usando aritmética de ponto flutuante de 32 bits. Os nós de quantização de peso (“wt quant”) e quantização de ativação (“act quant”) são injetados no gráfico de computação para simular os efeitos da quantização das variáveis ​​(seção 3). O gráfico resultante aproxima-se dos gráficos de computação somente com aritmética inteira no painel a), sendo treinável usando algoritmos de otimização eficientemente para modelos de ponto flutuante. c) Nosso esquema de quantização se beneficia dos circuitos rápidos de aritmética inteira em CPUs comuns para fornecer uma melhor relação entre latência e precisão (seção 4). A figura comparaMobileNets quantizados em inteiros [10] com linhas de base de ponto flutuante no ImageNet [3] usando núcleos Qualcomm Snapdragon 835 LITTLE.

Eles se inspiram em 2 artigos:

- O primeiro eles aproveitam aritmética de ponto fixo de baixa precisão para acelerar o treinamento de CNNs 
- O segundo usa aritmética de ponto fixo de 8 bits para acelerar a inferência em CPUs x86

## 2. Quantized Inference

### 2.1. Quantization scheme

Ele usam quantização linear com parametrização assimétrica

Segue o mesmo esquema base de operadores que eu com diferença na sintexe de C++, por permitir transferência de tipo mais tranquilamente.

O desafio deles é como realizar inferência usando apenas aritmética inteira para traduzir a computação de números reais em computação de valores quantizados e como projetar o sistmea para envolver apenas aritmética inteira, mesmo que os valores de escala não sejam inteiros.

### 2.2. Integer-arithmetic-only matrix multiplication

$$q_3^{(i,k)} = Z_3 + M \sum_{j=1}^{N} (q_1^{(i,j)} - Z_1)(q_2^{(j,k)} - Z_2)$$

Ele comenta sobre a multiplicação de matrizes e aqui tem um ponto importante. Quando o texto diz que o multiplicador $M=\frac{S1S2}{S3}$ pode ser calculado offline”, significa que ele não precisa ser computado durante a multiplicação real das matrizes, ou seja, não depende dos elementos da matriz, apenas das escalas de quantização de $S_1, S_2, S_3$ que já foram definidas ou calibradas previamente. Descobriram empiricamente que o valor de M fica sempre entre $(0,1)$, então fui pesquisar como.

A escala $S$ representa o "tamanho" de um degrau na quantização. Se $S$ for muito grande, a precisão é baixa. Em redes neurais, as ativações (o resultado da multiplicação, $S_3$) geralmente precisam ser capazes de representar uma amplitude similar ou maior que os valores de entrada ($S_1$ e $S_2$). Se $M$ fosse muito maior que $1$, cada multiplicação de matriz faria os números crescerem exponencialmente camada após camada, causando um overflow nos inteiros. Para manter os valores sob controle e aproveitar a precisão dos bits, os engenheiros calibram as escalas de modo que $M$ seja pequeno. Na prática, se $M$ for maior que $1$, você pode simplesmente aumentar o valor de $n$ na fórmula para "puxá-lo" de volta para o intervalo desejado. Então eles usam uma maneira de representar esse valor,

$$M = M_0 \times 2^{-n}$$

Imagine que você quer multiplicar um número por $0,00723$, mas seu computador não sabe o que é vírgula. Ele só entende números inteiros. Como você faz? Você transforma o problema em uma notação científica binária. Pense em base 10 primeiro. O número $0,00723$ pode ser escrito como:

$$7,23 \times 10^{-3}$$

Aqui, $7,23$ é o seu "$M_0$" e $-3$ é o seu "$n$". 

O hardware prefere a base 2. Então fazemos: 
- $M_0$ (A Mantissa): É um valor normalizado entre $[0,5, 1)$. Nós o transformamos em um inteiro gigante (usando 31 bits) para não perder nenhuma casa decimal.
- $2^{-n}$ (O Shift): É o ajuste da vírgula. Multiplicar por $2^{-n}$ é exatamente a mesma coisa que empurrar os bits $n$ vezes para a direita.

Se $M_0 = 0,99$, o valor inteiro será quase $2^{31}$. Se $M_0$ pudesse ser um valor muito pequeno, tipo $0,00001$, o valor inteiro seria pequeno e sobrariam muitos zeros à esquerda "vazios". Ao forçar $M_0$ a ser sempre $\geq 0,5$, garantimos que o primeiro bit (ou o segundo, dependendo do sinal) sempre será 1. Isso significa que estamos usando todos os 30 ou 31 bits disponíveis para guardar os detalhes do número. É como esticar uma imagem para que ela ocupe a tela inteira em vez de ficar um quadradinho minúsculo no canto.

Obs.:
- Offline = antes da execução real do algoritmo, durante a fase de preparação ou compilação.
- Online = durante a execução da multiplicação real, usando os valores das matrizes.

Isso economiza cálculos repetidos. Você só calcula $M$ uma vez e permite otimizações de hardware/software, como:
- Armazenar $M$ como inteiro em ponto fixo
- Reduzir multiplicações de ponto flutuante durante a execução

#### 2.3. Efficient handling of zero-points

Imagine que você tem duas matrizes quadradas de tamanho $N \times N$ (por exemplo, $1000 \times 1000$). Para multiplicá-las, o algoritmo padrão usa três loops aninhados (um dentro do outro):
- Um para as linhas da matriz A.
- Um para as colunas da matriz B.
- Um para percorrer os elementos e fazer a soma dos produtos.

Se cada loop roda $N$ vezes, o total de operações é $N \times N \times N = \mathbf{N^3}$. Se $N=10$, são $1.000$ operações. Se $N=1000$, são 1 bilhão de operações. Qualquer coisinha "boba" que você coloca dentro do loop mais interno será repetida bilhões de vezes.

O que é o $2N^3$ subtrações?

Na Equação (4), para cada uma das $N^3$ multiplicações, você teria que fazer:

- $(q_1 - Z_1)$ → 1ª subtração
- $(q_2 - Z_2)$ → 2ª subtração

Como essas duas subtrações ocorrem no coração do loop triplo, você acaba realizando $2 \times N^3$ subtrações. Em uma matriz de 1000x1000, seriam 2 bilhões de subtrações inúteis, já que os valores de $Z_1$ e $Z_2$ não mudam!

O objetivo é tirar essas subtrações do loop $N^3$.

As Somas de Linha ($\bar{a}_1^{(i)}$) e Coluna ($a_2^{(k)}$): Elas são calculadas com apenas dois loops ($N \times N$). Isso é chamado de $O(N^2)$.

O termo $Z_2 \bar{a}_1^{(i)}$ é calculado apenas uma vez para cada posição da matriz resultante ($N \times N$). 

A diferença de esforço para o processador é brutal. Antes o processador fazia a multiplicação e as subtrações no nível $N^3$ e depois passou a fazer a multiplicação pura no nível $N^3$ e deixa as subtrações/correções para o nível $N^2$. É a diferença entre você carregar 10 tijolos por vez subindo 1000 degraus ($N^3$), ou subir os 1000 degraus leve e só no final encontrar os tijolos lá em cima ($N^2$).

Na Equação (4), para cada elemento da matriz resultante, você teria que fazer:

$$(q_1^{(i,j)} - Z_1) \times (q_2^{(j,k)} - Z_2)$$

Isso significa que, para cada par de números sendo multiplicados, você precisa subtrair os *Zero Points*. Se sua matriz for grande, você fará bilhões dessas subtrações. Além disso, ao subtrair, o número pode ficar negativo ou maior que 8 bits, forçando você a converter tudo para 16 bits antes de multiplicar.

O texto sugere "distribuir a multiplicação". Se você expandir o produto , o resultado é . Aplicando isso aos termos da somatória:

$$\sum (q_1 - Z_1)(q_2 - Z_2) = \sum (q_1 q_2 - q_1 Z_2 - Z_1 q_2 + Z_1 Z_2)$$

Ao separar essa somatória em quatro partes independentes, chegamos à segunda fórmula da imagem:

1. $\sum q_1 q_2$: Esta é a multiplicação de matrizes pura entre os valores inteiros. É o que o hardware faz de melhor.
2. $- Z_2 \bar{a}_1^{(i)}$: Aqui, $\bar{a}_1^{(i)}$ é apenas a soma da linha $i$ da primeira matriz.
3. $- Z_1 a_2^{(k)}$: Aqui, $a_2^{(k)}$ é apenas a soma da coluna $k$ da segunda matriz.
4. $N Z_1 Z_2$: Como $Z_1, Z_2$ e $N$ são constantes, esse valor é um número fixo.

A grande vantagem é que os termos 2, 3 e 4 podem ser calculados "fora" do loop principal ou de forma muito mais rápida:

* **Somas de Linhas/Colunas:** Você só precisa somar os elementos de cada linha da matriz  e cada coluna da matriz  **uma única vez**. Isso custa .
* **O Loop Principal ($O(N^3)$):** Agora, dentro do loop triplo da multiplicação de matrizes, o processador só precisa fazer `soma += q1 * q2`. Não há mais subtrações de  ali dentro.
* **Ajuste Final:** As compensações (os termos com  e ) são aplicadas apenas uma vez por elemento da matriz resultante, e não para cada multiplicação individual.

Basicamente, eles transformaram um problema de "ajustar cada número antes de multiplicar" em um problema de "multiplicar tudo direto e corrigir o erro no final usando as somas das linhas e colunas".

### 5. Discussion

Foi proposto um esquema de quantização que se baseia apenas em aritmética de inteiros para aproximar os cálculos de ponto flutuante em uma rede neural. O treinamento que simula o efeito da quantização ajuda a restaurar a precisão do modelo a níveis quase idênticos aos originais. Além da redução de 4 vezes no tamanho do modelo, a eficiência da inferência é aprimorada por meio de implementações baseadas em ARM NEON.


Para habilitar operações com números inteiros em uma rede neural de ponto flutuante pré-treinada requer duas operações fundamentais:

- Quantizar: converter um número real em uma representação inteira quantizada (por exemplo, de fp32 para int8).
- Desquantizar: converter um número de uma representação inteira quantizada para um número real (por exemplo, de int32 para fp16).

Vamos pensar em um intervalo real qualquer, exemplo [-1.2, 2.3]:

$[\beta, \alpha]$

Uma representação inteira com $b$ bits com sinal. Isso significa que os inteiros possíveis são:

$[−2^{b−1}, 2^{b−1}−1]$

Exemplos:
- int8 -> $[−2^{8−1}, 2^{8−1}−1] -> [−2^{7}, 2^{7}−1] [−128, 127]$
- int4 -> $[−2^{4−1}, 2^{4−1}−1] -> [−2^{3}, 2^{3}−1] [−8, 7]$
- int2 -> $[−2^{2−1}, 2^{2−1}−1] -> [−2, 2−1] [−2, 1]$
- int1 -> $[−2^{1−1}, 2^{2−1}−1] -> [−1, 0] [−128, 127]$