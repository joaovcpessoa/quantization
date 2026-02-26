# INTEGER QUANTIZATION FOR DEEP LEARNING INFERENCE: PRINCIPLES AND EMPIRICAL EVALUATION

## Abstract

As técnicas de quantização podem reduzir o tamanho das Redes Neurais Profundas e melhorar a latência e a taxa de transferência da inferência, aproveitando as instruções de alta taxa de transferência para operações com números inteiros. Neste artigo, revisamos os aspectos matemáticos dos parâmetros de quantização e avaliamos suas escolhas em uma ampla gama de modelos de redes neurais para diferentes domínios de aplicação, incluindo visão computacional, fala e linguagem. Nosso foco são as técnicas de quantização que podem ser aceleradas por processadores com pipelines de matemática de números inteiros de alta taxa de transferência. Também apresentamos um fluxo de trabalho para quantização de 8 bits que consegue manter a precisão dentro de 1% da linha de base de ponto flutuante em todas as redes estudadas, incluindo modelos mais difíceis de quantizar, como MobileNets e BERT-large.

## Introduction

Embora o formato numérico dominante para aplicações de Aprendizado Profundo (DL) fosse o de ponto flutuante de precisão simples de 32 bits, recentemente diversos formatos alternativos foram propostos para aumentar o desempenho computacional dessas aplicações. Está se tornando comum treinar redes neurais em formatos de ponto flutuante de 16 bits, como IEEE fp16 [35] ou bfloat16 [57], suportados pela maioria dos aceleradores de DL. Uma vez treinadas, as redes neurais podem ser utilizadas para inferência em formatos de precisão ainda menor, incluindo ponto flutuante, ponto fixo e inteiro. Os formatos de baixa precisão oferecem diversas vantagens de desempenho. Primeiro, muitos processadores fornecem pipelines matemáticos de maior taxa de transferência para os formatos de baixa precisão, o que pode acelerar operações matemáticas intensivas, como convoluções e multiplicações de matrizes. Segundo, tamanhos de palavra menores reduzem a pressão sobre a largura de banda da memória, melhorando o desempenho em computações com largura de banda limitada. Terceiro, tamanhos de palavra menores levam a menores requisitos de memória, o que pode melhorar a utilização do cache, bem como outros aspectos da operação do sistema de memória.

Neste artigo, focamos na quantização inteira para inferência em redes neurais, onde as redes treinadas são modificadas para usar pesos e ativações inteiras, de modo que pipelines matemáticos inteiros possam ser usados ​​para muitas operações. A Tabela 1 lista as taxas de transferência relativas de operações tensoriais de vários tipos de dados na arquitetura de Unidade de Processamento Gráfico (GPU) NVIDIA Turing [40]. Operações tensoriais com uso intensivo de matemática, executadas em tipos inteiros de 8 bits, podem apresentar um aumento de velocidade de até 16 vezes em comparação com as mesmas operações em fp32. Operações com memória limitada podem apresentar um aumento de velocidade de até 4 vezes em comparação com a versão fp32, devido ao menor tamanho da palavra. Outros processadores, como TPUv1 [23], CPUs Intel com instruções VNNI [28] e diversos projetos de aceleradores emergentes também fornecem aceleração significativa para operações int8. O processo de quantização de redes neurais pode ser automatizado por ferramentas de software [36, 61] ou controlado manualmente. Em ambos os casos, deve-se tomar cuidado para minimizar qualquer impacto que a quantização tenha na precisão do modelo.

![alt text](image000.png)

Neste artigo, revisamos os fundamentos matemáticos subjacentes a diversas opções de quantização inteira (Seção 3), bem como técnicas para recuperar a precisão perdida devido à quantização (Seção 5). A Seção 6 combina essas informações em um fluxo de trabalho recomendado. Na Seção 4 e nos Apêndices, apresentamos uma avaliação empírica de diversas opções de quantização em uma ampla gama de modelos de redes de diferentes domínios de aplicação — processamento de imagens, modelagem de linguagem, tradução de idiomas e reconhecimento de fala. Esses modelos incluem as principais topologias de rede — redes convolucionais, redes recorrentes e redes baseadas em atenção. Com o fluxo de trabalho apresentado para quantização int8, conseguimos manter a precisão do modelo dentro de 1% de cada rede de ponto flutuante de referência, mesmo para as redes que são conhecidas por serem difíceis de quantizar, como MobileNets e BERT-large.

## Related Work

Vanhoucke et al. [52] showed that earlier neural networks could be quantized after training to use int8 instructions on Intel CPUs while maintaining the accuracy of the floating-point model. More recently it has been shown that some modern networks require training to maintain accuracy when quantized for int8. Jacob et al. [20] described models optimized for inference where all inference operations were performed with integer data types. Here batch normalization layers were folded into the preceding convolution layer before quantization, reducing the number of layers that needed to be executed during inference. Krishnamoorthi [26] evaluated various quantization methods and bit-widths on a variety of Convolutional Neural Networks (CNNs). He showed that even with per-channel quantization, networks like MobileNet do not reach baseline accuracy with int8 Post Training Quantization (PTQ) and require Quantization Aware Training (QAT). McKinstry et al. [33] demonstrated that many ImageNet CNNs can be finetuned for just one epoch after quantizing to int8 and reach baseline accuracy. They emphasized the importance of using an annealing learning rate schedule and a very small final learning rate. They also set the quantization range based on a percentile of activations sampled from the training set. Instead of using fixed ranges, Choi et al. [6] proposed PACT which learns the activation ranges during training.

Much of the earlier research in this area focused on very low bit quantization [7, 13, 59], all the way down to ternary (2-bit) [60, 34] and binary weights [8] and activations [45, 18]. These works showed that for lower bit-widths, training with quantization was required to achieve high accuracy, though accuracy was still lower than the floating-point network on harder tasks such as ImageNet image classification [47]. They also demonstrated the importance of techniques such as using higher precision for weight updates and the Straight-through Estimator (STE) for gradient backpropagation [3]. Also, in many cases the first and last layer were not quantized, or quantized with a higher bit-width, as they are more sensitive to quantization [59, 45, 18]. Multi-bit quantization schemes use either uniform [7, 59], or non-uniform quantization [13, 60, 34, 2]. Uniform quantization enables the use of integer or fixed-point math pipelines, allowing computation to be performed in the quantized domain. Non-uniform quantization requires dequantization, e.g. a codebook lookup, before doing computation in higher precision, limiting its benefits to model compression and bandwidth reduction. This paper focuses on leveraging quantization to accelerate computation, so we will restrict our focus to uniform quantization schemes.

While much of the aforementioned work has focused on CNNs for image classification, there are also many examples of applying quantization to other types of network architectures. Wu et al. [55] described how Google’s Neural Machine Translation (GNMT), which employs a Long Short Term Memory (LSTM) Recurrent Neural Network (RNN), was trained with hard range constraints on multiple tensors to be more amenable to PTQ. A similar strategy was taken on MobileNet v2 [48], which restricts activations to be in the range [0, 6] (ReLU6). Bhandare et al. [4] quantized the smaller base Transformer [53] model targeting the int8 VNNI instructions on Intel CPUs. They use KL-Divergence [36] to calibrate the quantization ranges and apply PTQ. Zafrir et al. [58] quantized BERT [10] to int8 using both PTQ and QAT. In this paper, we present an evaluation of int8 quantization on all of the major network  architectures with both PTQ and QAT.

More complex methods have also been proposed for training quantized models. Distillation has been used to train a quantized “student” model with a high precision, and often larger, “teacher” model. It has been applied to training quantized CNNs [37, 43], LSTMs [43] and Transformers [24]. Leng et al. [31] used the Alternating Direction Method of Multipliers (ADMM) as an alternative to STE when training quantized model. These methods generally target lower bit-width quantization, as QAT has been shown to be sufficient for int8 quantization. We have also found QAT to be sufficient for int8 quantization on the models we evaluated, and as such we chose not to included these methods in our evaluation of int8 quantization.

![alt text](fig1.png)

Figure 1: Quantization mapping of real values to int8

## Quantization Fundamentals

We focus on uniform integer quantization as it enables computing matrix multiplications and convolutions in the integer domain, allowing the use of high throughput integer math pipelines. Uniform quantization can be divided in to two steps. First, choose the range of the real numbers to be quantized, clamping the values outside this range. Second, map the real values to integers representable by the bit-width of the quantized representation (round each mapped real value to the closest integer value).

In this Section we will consider higher precision floating-point formats like fp16 and fp32 to be real numbers for the purpose of discussion. Enabling integer operations in a pre-trained floating-point neural network requires two fundamental operations:
- Quantize: convert a real number to a quantized integer representation (e.g. from fp32 to int8).
- Dequantize: convert a number from quantized integer representation to a real number (e.g. from int32 to fp16).

We will first define the quantize and dequantize operations in Section 3.1 and discuss their implications in neural network quantization in Sections 3.2 and 3.3. Then we will discuss how the real ranges are chosen in Section 3.4.

### Range Mapping

Let $[β, α]$ be the range of representable real values chosen for quantization and b be the bit-width of the signed integer representation. Uniform quantization transforms the input value $x ∈ [β, α]$ to lie within $[−2^{b−1}, 2^{b−1}-1]$, where inputs outside the range are clipped to the nearest bound. Since we are considering only uniform transformations, there are only two choices for the transformation function: $f(x) = s · x + z$ and its special case $f(x) = s · x$, where $x, s, z ∈ R$. In this paper we refer to these two choices as affine and scale, respectively.

#### Affine Quantization

Affine quantization maps a real value $x ∈ R$ to a $b$-bit signed integer $x_q Equations 1 and 2 define affine transformation function, $f(x) = s · x + z$:

$$s = \frac{2^b - 1}{α − β}$$ (1)
$$z = − round(β · s) − 2^{b−1}$$ (2)

where $s$ is the scale factor and $z$ is the zero-point - the integer value to which the real value zero is mapped. In the 8-bit case, $s = \frac{255}{α − β}$ and $z = −round(β · s) − 128$. Note that $z$ is rounded to an integer value so that the real value of zero is exactly representable. This will result in a slight adjustment to the real representable range $[β, α]$ [20].

The quantize operation is defined by Equation 3 and 4:

### Quantization-Aware Training

Quantization Aware Training (QAT) describes the technique of inserting quantization operations in to the neural network before training or fine-tuning, to allow the network to adapt to the quantized weights and activations. Appendix B illustrates how this can lead to a better result. We apply QAT to fine-tuning as it has been shown that starting from a pre-trained network and fine-tuning leads to better accuracy [37, 26] and requires significantly fewer iterations [33].
This also allows us to leverage the calibrated pre-trained models from Section 4. Note that we keep the quantization ranges fixed throughout fine-tuning. Another approach is to learn the ranges, which we evaluate in Section 5.3.

A common approach to implementing QAT is to insert fake quantization, also called simulated quantization [26], operations into a floating-point network. Equation 12 defines fake quantization as a quantize and dequantize operation that produces an approximate version of the input, xˆ ≈ x, where x and xˆ are both floating-point values.


![alt text](tab7.png)

![alt text](tab8.png)

### Learning Quantization Parameters

While the techniques described in the previous sections relied on quantization parameters calibrated on the pre-trained network, it is also possible to jointly learn the quantization parameters along with the model weights. PACT [6] proposed learning the ranges for activation quantization during training. In this section we adopt PACT as an enhancement to our quantization aware fine-tuning procedure. We follow the same fine-tuning schedule as before, described in Appendix A, but allow the ranges of each quantized activation tensor to be learned along with the weights, as opposed to keeping them fixed throughout fine-tuning.

Table 8 shows a selection of networks fine-tuned with fixed and learned activation ranges for different initial calibrations. The “best” calibration refers to the calibration that produced the best accuracy with PTQ, as shown in Table 5. When the activation quantization is initialized with max calibration, learning the range results in higher accuracy than keeping it fixed for most networks. In particular it results in substantial accuracy improvements where fixed max ranges resulted in a significant accuracy drop. However, when activation ranges are initialized the to the best calibration for each network, learning the ranges yield very similar results to fixed ranges. This suggests that learning the ranges does not offer additional benefit for int8 over QAT if activation ranges are already carefully calibrated. However, this may not be the optimal application of PACT. Comparing the learned range results on Inception v4 suggest that when starting from max, the network was not able to learn a good activation ranges in the given fine-tuning schedule. We expect that PACT would be able to learn a better range with longer fine-tuning, or a separate optimization schedule and hyperparameters for the range parameters, such and learning rate and weight decay.

![alt text](fig5.png)

## Recommended Workflow

Based on the results in Sections 4 and 5, we recommend the following for int8 quantization:

- Weights:
    - Use scale quantization with per-column/per-channel granularity
    - Use a symmetric integer range for quantization $[-127, 127]$ and max calibration
- Activations:
    - Use scale quantization with with per-tensor granularity

We recommend the following procedure to quantize a pre-trained neural network.

- PTQ: Quantize all the computationally intensive layers (convolution, linear, matrix multiplication, etc.) and run activation calibration including max, entropy and 99.99%, 99.999% percentile. If none of the calibrations yield the desired accuracy continue to partial quantization or QAT.
- Partial Quantization: Perform sensitivity analysis to identify the most sensitive layers and leave them in floating-point. If the impact on computational performance is not acceptable or an acceptable accuracy cannot be reached, continue to QAT.
- QAT: Start from the best calibrated quantized model. Use QAT to fine-tune for around 10% of the original training schedule with an annealing learning rate schedule starting at 1% of the initial training learning rate.
Refer to Appendix A.2 for specific hyperparameter choices.

Figure 5 summarizes the above workflow in a flowchart.

## Conclusions

This paper reviewed the mathematical background for integer quantization of neural networks, as well as some performance-related reasons for choosing quantization parameters. We empirically evaluated various choices for int8 quantization of a variety of models, leading to a quantization workflow proposal. Following this workflow we demonstrated that all models we studied can be quantized to int8 with accuracy that either matches or is within 1% of the floating-point model accuracy. This included networks that are challenging for quantization, such as MobileNets and BERT. The workflow involves only post-training quantization, partial quantization, and quantization-aware fine-tuning techniques. Some more complex techniques, such as ADMM and distillation, were not required for int8 quantization of these models. However, these techniques should be evaluated when quantizing to even lower-bit integer representations, which we leave to future work.









