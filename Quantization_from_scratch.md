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

# Quantization from Scratch

## Summary

1. Model Compression Techniques
    - Pruning
    - Quantization
    - Knowledge Distillation
    - Low-Rank Decomposition
    - Joint Compression Methods

2. Foundations of Quantization
    - 2.1. Concept
        - Continuous-to-discrete mapping
        - Information loss and distortion
        - Quantization noise interpretation
    - 2.2. Perspectives
        - Linear Interpolation Perspective
        - DSP Perspective
        - Machine Learning Perspective

3. Quantization Modeling Space
    - 3.1. Mapping Function
        - Linear Quantization
        - Nonlinear Quantization
            - Logarithmic
            - Power-of-Two
            - Learned Nonlinear Mapping
    - 3.2. Numerical Parameterization
        - Symmetric Parameterization
        - Asymmetric Parameterization
        - Zero-Point Modeling
        - Scale Modeling
        - Rounding vs. Truncation
        - Stochastic Rounding
    - 3.3. Granularity Strategy
        - Per-Tensor
        - Per-Channel
        - Per-Group
        - Block-wise
        - Head-wise (Transformers)
    - 3.4. Bit-Width Strategy
        - Fixed Precision
        - Mixed Precision
        - Ultra-Low Bit (≤4-bit)
        - Layer-wise Bit Allocation
        - Differentiable Bit-Width Optimization
    - 3.5. Quantization Scope
        - Weights Only
        - Activations Only
        - Weights + Activations
        - Bias Quantization
        - Attention-Specific Quantization (Q/K/V)
    - 3.6. Structural Graph Transformations
        - BatchNorm Folding
        - Layer Fusion
        - Bias Correction
        - Operator Reordering
        - Scale Propagation

4. Quantization Optimization Strategies
    - 4.1. Post-Training Quantization (PTQ)
        - 4.1.1. Static PTQ (Calibration-Based)
            - Calibration Strategy
            - Min–Max
            - Histogram-Based
            - KL-Divergence
            - MSE-Based
            - Percentile Clipping
            - Optimized/Learned Clipping
        - 4.1.2. Dynamic PTQ
            - Offline Weight Quantization
            - Runtime Activation Quantization
            - Per-Tensor Scaling
            - Target Layers
                - Linear
                - LSTM
                - Transformer Blocks
        - 4.1.3. Layer-wise/Selective PTQ
            - Sensitive Layer Preservation (FP32)
            - Robust Layer Quantization
            - First/Last Layer Exemption
            - Automated Bit Allocation
            - Hardware-Driven Partitioning
                - CPU
                - GPU
                - NPU/Accelerator
        - 4.1.4. Modern PTQ Extensions
            - Data-Free/Zero-Shot PTQ
                - Synthetic Calibration Data
                - Generative Calibration
                - No Original Dataset
            - LLM-Oriented PTQ
                - Weight-Only Quantization
                - Activation-Aware Quantization
                - Block-wise Quantization
                - Group-wise Scaling
                - Outlier Channel Handling
                - Hessian-Based Methods
            - Joint Compression
                - Quantization + Pruning
                - Quantization + Knowledge Distillation
                - Quantization + Low-Rank Decomposition
    - 4.2 Quantization-Aware Training (QAT)
        - 4.2.1 Quantization Simulation Mechanisms
            - Fake Quantization Nodes
            - Straight-Through Estimator (STE)
            - Quantization Noise Injection
            - Differentiable Approximation
        - 4.2.2 Training Strategy Variants
            - Full QAT
            - Fine-Tuning QAT
            - Progressive / Partial QAT
            - Curriculum Quantization
            - Staged Bit-Width Reduction
        - 4.2.3. Learnable Quantization
            - Trainable Scale Parameters
            - Learnable Clipping Ranges
            - Learnable Zero-Points
            - Learned Step Size Quantization
            - PACT-style Clipping
            - 4.2.4. Precision Strategy in QAT
                - INT8 QAT
                - INT4 QAT
                - Binary/Ternary Networks
                - Mixed Precision QAT
                - INT8 + FP16
                - INT8 + FP32
                - Layer-wise Bit-Width Search
                - Differentiable Bit-Width Optimization
            - 4.2.5. Structural & Regularization Techniques
                - BatchNorm Folding (post-training)
                - Layer Fusion
                - Weight Clipping
                - Range Regularization
                - Quantization-aware Weight Decay
            - 4.2.6 Deployment Alignment
                - Hardware-aware QAT
                - Operator Constraint Matching
                - Backend-specific Rules
                - Accelerator-Constrained Training
                - Latency-aware Quantization
                - Energy-aware Optimization

5. Systems & Deployment Considerations
    - Integer-only Inference
    - Quantized Operator Libraries
    - Backend Rule Matching
    - Graph Rewriting for Inference
    - Compiler Integration
    - Runtime Scaling Policies

## Model Compression Techniques

Before discussing quantization and its techniques, we need to address why model compression has become a central topic in modern machine learning. Deep learning models have grown explosively in terms of parameter count, computational cost, and energy consumption. While this growth brings performance gains, it also introduces practical barriers: high latency, large memory footprints, reliance on specialized hardware, and difficulties deploying models on resource-constrained devices such as smartphones, embedded systems, and microcontrollers.

In practice, we rarely train models solely to run on data-center GPUs. The real challenge begins when we need to perform inference efficiently, predictably, and at low cost. This is where compression techniques come into play: they aim to reduce computational cost and model footprint without degrading or with only minimal degradation of accuracy. From both a mathematical and an engineering perspective, this requires rethinking how numbers are represented, how capacity is distributed across the network, and which components of the model are truly necessary.

There are three main pillars for making a model more efficient:

- **Quantization**: representing model parameters and/or activations with reduced precision (for example, from float32 to int8, or even more constrained formats)

- **Knowledge Distillation**: training a smaller “student” model using the outputs (or probability distributions) of a larger “teacher” model

- **Pruning**: removing unnecessary connections (weights) within the model by exploiting redundancy and inducing sparsity

Although these techniques are complementary, the focus here will be on quantization, particularly from the perspective of fixed-point arithmetic. Rather than treating quantization as a framework-provided “black box,” the goal is to open the hood: to understand how real numbers are approximated, what errors are introduced, how these errors propagate, and how to design efficient representations in a controlled manner. The objective is to build custom tools and sufficient mathematical intuition to go beyond theory and beyond off-the-shelf APIs.

## Quantization Concept

Quantization is the process of mapping a large (often continuous) set of values to a smaller, discrete set. In the context of neural networks, we typically focus on linear quantization, which uses a linear mapping to convert high-precision values (such as *float32*) into lower-precision representations (such as *int8*).

The idea of quantization long predates deep learning. It originates in signal processing and digital communications, where continuous signals must be represented using a finite number of bits in order to be stored, transmitted, or processed by digital hardware. In this setting, quantization is an unavoidable step: once we move from the analog to the digital world, real numbers can no longer be represented exactly and must be approximated.

Modern neural networks inherit this same constraint. While training is usually performed in floating-point arithmetic for numerical stability and optimization convenience, inference ultimately runs on digital hardware with finite precision. Historically, CPUs and embedded processors were designed around fixed-point or low-precision integer arithmetic, and even today, accelerators such as GPUs, TPUs, NPUs, and microcontrollers achieve their highest throughput and energy efficiency when operating on low-precision data types.

From this perspective, quantization is not merely a compression trick, it is a way of aligning mathematical models with the realities of hardware. By reducing numerical precision, we reduce memory usage, memory bandwidth, and computational cost, often by large factors. For example, replacing *float32* weights with int8 values reduces storage by $4 \times$ and enables the use of highly optimized integer matrix multiplication units.

Linear quantization, in particular, strikes a balance between simplicity and effectiveness. It assumes that real values can be approximated by uniformly spaced discrete levels, defined by a scale factor (and, in some cases, a zero-point). This structure makes quantized computations easy to implement in hardware and easy to reason about analytically, which is why it is the dominant approach in practical systems.

However, quantization is inherently lossy. Mapping continuous values to a discrete grid introduces quantization error, and understanding how this error arises, how it propagates through layers, and how it interacts with nonlinearities is central to designing robust quantized models. Rather than treating quantization as a purely empirical or framework-driven procedure, we will approach it as a controlled approximation problem, one that sits at the intersection of numerical analysis, signal processing, and machine learning.

## Quantization Approaches

Quantization can be described from different conceptual viewpoints, depending on the application domain. In this section, I would like to discuss three complementary perspectives: a **linear interpolation-based formulation**, a **classical digital signal processing (DSP) interpretation**, and a **machine learning-oriented view**.

### Linear Interpolation Perspective

#### Linear Quantization

Given a continuous-valued signal $x \in [x_{\min}, x_{\max}]$
and a bit-width of $b$ bits, linear quantization can be formally defined as follows.

##### Number of Quantization Levels

$L = 2^b$

##### Quantization Step Size

$\Delta = \frac{x_{\max} - x_{\min}}{L - 1}$

##### Quantization Operator

$Q_{\text{linear}}(x) =
\Delta \cdot \operatorname{round}!\left(\frac{x - x_{\min}}{\Delta}\right) + x_{\min}$

In the implementation under consideration, the signal range is normalized as:

$x_{\min} = -1, \quad x_{\max} = 1$

**Where:**
* $L$: Total number of available quantization levels (e.g., 256 for 8-bit quantization).
* $b$: Number of bits used for representation (bit depth).
* $\Delta$: Quantization step size; the distance between consecutive quantization levels.
* $x_{\min}, x_{\max}$: Minimum and maximum values of the original signal (full-scale range).
* $x$: Original continuous-valued input.
* $Q_{\text{linear}}(x)$: Quantized and reconstructed output value.
* $\operatorname{round}(\cdot)$: Rounding to the nearest integer.

This formulation can be interpreted as a **piecewise-constant linear interpolation**, where each input sample is mapped to the nearest discrete level on a uniformly spaced grid.

### DSP Perspective

From a classical DSP perspective, quantization is analyzed primarily in terms of **signal amplitude range**, **quantization noise**, and **signal-to-noise ratio (SNR)**.

#### Signal Model and Quantizer Definition

Assuming a uniformly quantized signal with ( b ) bits and a symmetric full-scale range:

$$x[n] \in [-X_m, +X_m]$$

The number of quantization levels is:

$$L = 2^b$$

The quantization step size is given by:

$$\Delta = \frac{2 X_m}{L}$$

A mid-rise uniform quantizer can be expressed as:

$$Q(x[n]) = \Delta \left\lfloor \frac{x[n]}{\Delta} + \frac{1}{2} \right\rfloor
$$

where $\lfloor \cdot \rfloor$ denotes the floor operator, and the offset $\frac{1}{2}$ emulates rounding to the nearest level.

#### Quantization Error Model

The quantization error (or quantization noise) is defined as:

$$e[n] = x[n] - Q(x[n])$$

Under the standard high-resolution assumptions:
* the signal varies sufficiently fast relative to the quantization step
* the quantizer is not overloaded
* and ( x[n] ) spans many quantization levels

the error ( e[n] ) can be modeled as a random variable uniformly distributed over:

$$e[n] \sim \mathcal{U}\left(-\frac{\Delta}{2}, \frac{\Delta}{2}\right)$$

The mean and variance of the quantization noise are then:

$$\mathbb{E}{e[n]} = 0$$

$$\sigma_e^2 = \mathbb{E}{e^2[n]} = \frac{\Delta^2}{12}$$

#### Signal-to-Noise Ratio (SNR)

For a full-scale sinusoidal input:

$$x[n] = X_m \sin(\omega n)$$

the signal power is:

$$\sigma_x^2 = \frac{X_m^2}{2}$$

The resulting SNR is:

$$\mathrm{SNR} = \frac{\sigma_x^2}{\sigma_e^2}
= \frac{X_m^2 / 2}{\Delta^2 / 12}
= \frac{3}{2} \left( \frac{X_m}{\Delta} \right)^2$$

Substituting $\Delta = \frac{2 X_m}{2^b}$:

$$\mathrm{SNR} = \frac{3}{2} \cdot 2^{2b}$$

In decibels:

$$\mathrm{SNR}_{\mathrm{dB}} \approx 6.02,b + 1.76 \ \text{dB}$$

This expression highlights the well-known result that **each additional bit increases the SNR by approximately 6 dB** for a full-scale sinusoidal input.

#### Implications for Nonlinear Quantization

Uniform quantization yields an SNR that depends on the signal amplitude. For low-amplitude signals, the SNR degrades significantly. Nonlinear quantization schemes, such as μ-law and A-law companding, are designed to counteract this limitation by making the quantization step size approximately proportional to the signal magnitude. As a result, the effective SNR remains approximately constant over a wide dynamic range.

### Machine Learning Perspective

In machine learning systems, quantization is primarily applied to reduce **memory footprint, bandwidth, and computational cost**, especially during inference.

Two main signal categories are considered:

1. **Weights**
   Fixed model parameters learned during training. Their statistical distribution is often highly concentrated around zero, motivating the use of nonlinear or distribution-aware quantization schemes.

2. **Activations**
   Intermediate values propagated through the network layers during inference. Activation distributions are input-dependent and may require dynamic range calibration or per-layer scaling.

Unlike classical DSP, ML-oriented quantization is typically evaluated in terms of **model accuracy degradation**, **throughput**, and **hardware efficiency**, rather than purely signal fidelity metrics. Modern approaches often combine ideas from nonlinear quantization, clustering, and statistical modeling to better match the underlying data distributions.

In this context, our focus is the approach to Machine Learning.

## Quantization Mapping Types

To illustrate the difference between quantization types, linear quantization can be compared to a standard ruler, where all centimeter marks are evenly spaced. Nonlinear quantization, by contrast, resembles an “elastic” ruler, in which certain regions provide finer resolution than others.

### Linear Quantization

In linear quantization, quantization levels are uniformly distributed across the entire data range.

**Advantages**:<br>
Linear quantization is mathematically straightforward, easy to implement, and highly efficient for hardware execution. It primarily relies on scaling, rounding, and clipping operations, all of which are natively supported by vector instructions and specialized hardware units.

**Disadvantages**:<br>
When most values are concentrated near zero, a common scenario for neural network weights and activations, linear quantization wastes precision levels in rarely used extreme regions of the range. As a result, quantization error increases in the most relevant value regions, degrading overall representation accuracy.

### Nonlinear Quantization

In nonlinear quantization, the spacing between quantization levels is non-uniform. The core idea is to allocate more levels (or effective bits) to frequently occurring values and fewer levels to rare values, thereby minimizing the average quantization error. Common examples include:

- **Logarithmic Quantization**:<br>
Instead of applying a linear mapping, a logarithmic transformation is performed prior to quantization. This approach is widely used in audio signal processing, such as μ-law and A-law companding, because human hearing is significantly more sensitive to variations in low-intensity sounds than in high-intensity sounds.

- **Clustering-Based Quantization (K-Means)**:<br>
Model weights are grouped into *k* clusters, and each value is represented by its corresponding cluster centroid. While this method can achieve high precision, it requires a lookup table to map indices back to real values, which can increase latency and computational overhead.

- **Normal Float (NF4)**:<br>
Employed in the QLoRA algorithm for large language models, NF4 assumes that weights approximately follow a normal (Gaussian) distribution. Quantization levels are positioned so that each level has equal probability of occurrence, maximizing the statistical efficiency of the representation.

From a signal processing standpoint, linear quantization implies that each increment in the sampled digital value corresponds to a fixed-size increment in the analog domain. For example, an 8-bit analog-to-digital (ADC) or digital-to-analog (DAC) converter with an input range of $0$ to $1V$ provides a resolution of $1/256≈3.9 \ mV$ per bit, regardless of the actual signal amplitude.

In nonlinear quantization, logarithmic encoding schemes (such as μ-law or A-law) are typically used so that the quantization step size for small sample values is much smaller than that for large sample values. Ideally, the step size is approximately proportional to the signal amplitude. This results in an approximately constant signal-to-noise ratio (SNR), dominated by quantization noise, across a wide range of signal amplitudes. Another way to interpret this property is that fewer bits are required to achieve a target SNR within the amplitude range of interest.

## Quantization Parameterization

Once linear quantization is chosen as the mapping type, the next design decision concerns **how the real-valued range is aligned with the discrete integer grid**. This is determined by the *parameterization* of the quantizer, typically through a **scale** and, optionally, a **zero-point**.

Formally, linear quantization maps a real value $x \in \mathbb{R}$ to an integer $q \in \mathbb{Z}$ as

$q = \operatorname{clip}\left(\left\lfloor \frac{x}{s} \right\rceil + z, \ q_{\min}, \ q_{\max}\right)$

and dequantization recovers an approximation of the real value via

$\hat{x} = s \cdot (q - z)$

where:
* $s > 0$ is the **scale** (quantization step size),
* $z$ is the **zero-point**,
* $[q_{\min}, q_{\max}]$ defines the integer range (e.g., ([-128,127]) for int8),
* $\lfloor \cdot \rceil$ denotes rounding to the nearest integer.

The distinction between **symmetric** and **asymmetric** quantization lies entirely in how $z$ and the real-valued range are defined.

### Quantization with symmetric parameterization

The real value zero is mapped exactly to the integer zero ($z = 0$) and the representable real range is symmetric around zero:

$[-\alpha, +\alpha]$

where $\alpha > 0$ is typically chosen as

$\alpha = \max(|x_{\min}|, |x_{\max}|)$

Given an integer range $[q_{\min}, q_{\max}] = [-Q, Q] $ (e.g., ( Q = 127 ) for int8), the scale is

$s = \frac{\alpha}{Q}$

Thus, quantization reduces to

$q = \operatorname{clip}\left(\left\lfloor \frac{x}{s} \right\rceil, \ -Q, \ Q\right)$

and dequantization becomes

$\hat{x} = s \cdot q$

See how to construct an operator mathematically using NumPy:

```python
def linear_symmetric_quantize(
    tensor: np.ndarray, 
    scale: float,
    dtype: np.dtype = np.int8,
    mode: str = 'round') -> np.ndarray:
    
    tensor_fp = tensor.astype(np.float32)

    if mode == 'round':
        q = np.round(tensor_fp / scale)
    elif mode == 'truncate':
        q = np.fix(tensor_fp / scale)
    else:
        raise ValueError("mode should be 'round' or 'truncate'")
    
    qmin = -np.iinfo(dtype).max
    qmax = np.iinfo(dtype).max
    q = np.clip(q, qmin, qmax)

    return q.astype(dtype)

def linear_symmetric_dequantize(q_tensor: np.ndarray, scale: float) -> np.ndarray:
    return q_tensor.astype(np.float32) * scale

def get_q_scale(tensor: np.ndarray, dtype: np.dtype = np.int8):
    q_max = np.iinfo(dtype).max
    r_max_abs = np.max(np.abs(tensor))
    
    if r_max_abs < 1e-9:
        return 1.0
    
    scale = r_max_abs / q_max
    
    return float(scale)

def calculate_quantization_error(original: np.ndarray, dequantized: np.ndarray):
    mse = np.mean((original - dequantized) ** 2)
    
    signal_power = np.mean(original ** 2)
    noise_power = mse
    if noise_power == 0:
        return mse, float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return mse, snr
```

#### Properties

* **Exact zero representation**: ( x = 0 \Rightarrow q = 0 )
* **Uniform resolution** on both positive and negative sides
* **Simpler arithmetic**: no offset correction is required

If your tensor is `[0.0, 0.0, 0.0]`, the r_max_abs will be $0$:

$$s = \frac{0}{127} = 0$$

When quantizing $q = x / s$, division by zero results in `NaN` (Not a Number) or a runtime error. Once a `NaN` enters the network, it propagates and "destroys" all subsequent weights and activations. Therefore, it is customary to add a tiny value like $10^{-9}$ to the denominator, avoiding division by zero and underflow.

#### Practical implications

Symmetric quantization is especially well-suited for **weights**, whose distributions are usually centered around zero. It also aligns naturally with fixed-point arithmetic and hardware accelerators, where multiply–accumulate operations of the form

$$(s_w q_w) \cdot (s_a q_a)$$

can be efficiently implemented.

However, if the data distribution is significantly shifted away from zero, symmetric quantization may waste a substantial portion of the available integer range.

### Quantization with asymmetric parameterization

Asymmetric quantization relaxes the constraint that zero must map to zero. Instead, it allows an arbitrary real-valued interval:

$$[x_{\min}, x_{\max}]$$

to be mapped onto the integer range $[q_{\min}, q_{\max}]$.

The scale is defined as

$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$$

and the zero-point is chosen so that $x = 0$ is represented as accurately as possible:

$$z = \left\lfloor q_{\min} - \frac{x_{\min}}{s} \right\rceil$$

Quantization then follows:

$$q = \operatorname{clip}\left(\left\lfloor \frac{x}{s} \right\rceil + z, \ q_{\min}, \ q_{\max}\right)$$

and dequantization:

$$\hat{x} = s \cdot (q - z)$$

```python
def linear_asymmetric_quantize(
    tensor: np.ndarray, 
    scale: float, 
    zero_point: int, 
    dtype: np.dtype = np.int8,
    mode: str = 'round') -> np.ndarray:
    
    tensor_fp = tensor.astype(np.float32)
    
    if mode == 'round':
        q = np.round(tensor_fp / scale + zero_point)
    elif mode == 'truncate':
        q = np.fix(tensor_fp / scale + zero_point)
    else:
        raise ValueError("mode should be 'round' or 'truncate'")

    qmin = np.iinfo(dtype).min
    qmax = np.iinfo(dtype).max
    q = np.clip(q, qmin, qmax)

    return q.astype(dtype)

def linear_asymmetric_dequantize(q_tensor: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (q_tensor.astype(np.float32) - zero_point) * scale

def get_q_scale_and_zero_point(tensor: np.ndarray, dtype: np.dtype = np.int8, mode: str = 'round'):
    q_min = np.iinfo(dtype).min
    q_max = np.iinfo(dtype).max
    
    r_min = tensor.min()
    r_max = tensor.max()
    
    scale = (r_max - r_min) / (q_max - q_min)
    
    if scale == 0:
        scale = 1e-8
    
    zero_point = q_min - (r_min / scale)
    if mode == 'round':
        zero_point = np.clip(np.round(zero_point), q_min, q_max).astype(dtype)
    elif mode == 'truncate':
        zero_point = np.clip(np.fix(zero_point), q_min, q_max).astype(dtype)
    else:
        raise ValueError("mode should be 'round' or 'truncate'")
        
    return float(scale), zero_point
```

#### Properties
* Zero is **approximately** represented (exact only if $z$ is integer-aligned)
* The full integer range is used even for shifted distributions
* Better resolution for strictly non-negative signals

#### Practical implications

Asymmetric quantization is commonly used for **activations**, especially after ReLU or other non-negative nonlinearities. In these cases, forcing symmetry around zero would waste half of the representable levels.

The trade-off is increased arithmetic complexity: during inference, integer operations must compensate for the zero-point, which slightly increases computational overhead and implementation complexity.

### Interpretation as approximation error

In both cases, quantization introduces an approximation error:

$$\varepsilon(x) = x - \hat{x}$$

For linear quantization with rounding, this error is bounded by

$$|\varepsilon(x)| \le \frac{s}{2}$$

provided that $x$ lies within the representable range. The choice between symmetric and asymmetric parameterization directly affects **how this error is distributed** over the input domain.

### Rounding vs. Truncation in Quantized Neural Networks

Rounding minimizes the quantization error by selecting the nearest representable value. The maximum absolute error introduced by rounding is bounded by (0.5s), where (s) is the quantization scale. As a result, rounding provides higher numerical fidelity.

Truncation, defined as ($q = \lfloor x / s \rfloor$) (or truncation toward zero), discards the fractional component of the scaled value. Although truncation introduces a larger quantization error, it significantly reduces computational complexity and is therefore widely adopted in hardware-oriented implementations.

#### Accumulation and Precision Growth

Quantized linear and convolutional layers are implemented as sequences of **multiply–accumulate (MAC)** operations. The product of two INT8 operands yields a result of up to 15 bits plus sign. Accumulating a large number of such products would rapidly overflow narrow registers; consequently, hardware accelerators typically employ **INT32 accumulators** for intermediate results.

The choice between rounding and truncation becomes critical during the re-quantization of these INT32 accumulators back to INT8.

#### Re-quantization and Hardware Constraints

Re-quantization introduces a trade-off between numerical accuracy and hardware efficiency. While rounding reduces quantization error, it requires additional comparison or addition logic, increasing latency and hardware cost. Truncation, in contrast, can be implemented without auxiliary operations, enabling faster execution and reduced silicon area.

These considerations are particularly relevant in resource-constrained environments such as embedded systems, FPGAs, and ASIC accelerators.

#### Bit-Shifting and Scale Representation

When the quantization scale is a power of two, division by the scale can be implemented as a right bit shift. This operation inherently performs truncation by discarding the least significant bits. Implementing rounding in this context would require inspecting the discarded bits and conditionally adding a correction term, further increasing complexity.

As a result, truncation naturally aligns with efficient fixed-point arithmetic in hardware.

#### Effect on Network Dynamics

Quantization strategy can influence network behavior beyond numerical precision. Systematic rounding may introduce a small positive bias in activations, which can interact with non-linear functions such as ReLU by shifting values across the activation threshold. Truncation, by contrast, tends to limit activation magnitudes and may reduce unintended bias accumulation in deep networks.

In low-precision training scenarios, stochastic variants of truncation are often preferred to prevent small gradient values from consistently collapsing to zero.

## Quantization Granularity

Quantization granularity refers to the spatial or structural level at which quantization parameters—specifically the scaling factor  and the zero-point —are shared across a weight tensor or activation map. The choice of granularity represents a fundamental trade-off between the model's representational accuracy and the hardware efficiency of the resulting inference engine.

### Per-Tensor Quantization

In per-tensor (or layer-wise) quantization, a single set of quantization parameters is applied to the entire tensor. For a weight matrix , the quantized representation  is derived such that:

$$Q = \text{clip}\left(\lfloor \frac{W}{\Delta} \rceil + z, q_{min}, q_{max}\right)$$

where:
- $W$: original float value
- $\Delta$: scaling factor
- $z$: zero-point,
- $\lfloor \cdot \rceil$: rounding operator

While this approach minimizes memory overhead for storing metadata and simplifies the arithmetic units in hardware, it is highly susceptible to **outlier values**. A single high-magnitude element can skew the scaling factor, leading to significant rounding errors for the remaining values in the tensor.

Consider the previous code implementations by tensor. If you want, you can rename the functions by adding `_per_tensor`.

### Per-Channel Quantization

To mitigate the precision loss of per-tensor methods, per-channel quantization assigns independent scaling factors to each output channel or convolutional filter. This is formally expressed as:

$$\Delta_i = \frac{\max(|W_i|) - \min(|W_i|)}{2^b - 1}$$

where  denotes the channel index and  the bit-width. This granularity is particularly effective for Deep Neural Networks (DNNs) where weight distributions vary significantly between filters. It has become the standard for weight quantization in most production-grade frameworks, as it captures the inter-channel variance without significantly increasing computational complexity.

**What changes in the functions we create?**

#### Symmetric

<details><summary><b>python code</b></summary>

```python
def linear_symmetric_quantize_per_channel(
    tensor: np.ndarray, 
    scales: np.ndarray,
    axis: int = 0,
    dtype: np.dtype = np.int8,
    mode: str = 'round') -> np.ndarray:
    
    broadcast_shape = [1] * tensor.ndim
    broadcast_shape[axis] = -1
    s = scales.reshape(broadcast_shape)
    
    tensor_fp = tensor.astype(np.float32)

    if mode == 'round':
        q = np.round(tensor_fp / s)
    elif mode == 'truncate':
        q = np.fix(tensor_fp / s)
    else:
        raise ValueError("mode should be 'round' or 'truncate'")
    
    qmin = -np.iinfo(dtype).max
    qmax = np.iinfo(dtype).max
    q = np.clip(q, qmin, qmax)

    return q.astype(dtype)

def linear_symmetric_dequantize_per_channel(
    q_tensor: np.ndarray, 
    scales: np.ndarray,
    axis: int = 0) -> np.ndarray:
    
    broadcast_shape = [1] * q_tensor.ndim
    broadcast_shape[axis] = -1
    s = scales.reshape(broadcast_shape)
    
    return q_tensor.astype(np.float32) * s

def get_q_scale_per_channel(
    tensor: np.ndarray,
    axis: int = 0,
    dtype: np.dtype = np.int8):
          
    q_max = np.iinfo(dtype).max
    
    reduce_axes = tuple(i for i in range(tensor.ndim) if i != axis)
    r_max_abs = np.max(np.abs(tensor), axis=reduce_axes)
    
    scale = r_max_abs / q_max
    scale = np.where(scale < 1e-9, 1.0, scale)
    
    return scale

def calculate_quantization_error(original: np.ndarray, dequantized: np.ndarray):
    mse = np.mean((original - dequantized) ** 2)
    
    signal_power = np.mean(original ** 2)
    noise_power = mse
    if noise_power == 0:
        return mse, float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return mse, snr
```
</details>
<br>

The current implementation replaces global quantization parameters with per-channel (vectorized) parameters, allowing each channel in a neural network to operate at its own numerical resolution. This approach significantly improves precision, especially in layers where the statistical distribution of weights varies across channels.

1. Per-Channel Scale Computation (`get_q_scale_per_channel`)

Unlike per-tensor quantization, which derives a single scaling factor from the entire tensor, per-channel quantization computes independent statistics for each channel.

Channel-wise reduction:
The algorithm identifies all tensor dimensions except the channel axis and reduces them. For example, in a 4D convolution weight tensor, spatial dimensions and input depth are reduced while preserving the output channel dimension.

Maximum absolute value per channel (r_max_abs):
The maximum absolute value is computed independently for each channel. A tensor with 64 output channels will therefore produce 64 distinct maximum values.

Scale computation:
A separate scale is derived for each channel. Channels with smaller dynamic ranges receive smaller scales, which allows better utilization of the available integer range and preserves fine-grained numerical details.

This strategy mitigates range underutilization, a common issue in per-tensor quantization.

2. Broadcasting Mechanism

To efficiently apply per-channel scaling without explicit loops, the implementation relies on NumPy broadcasting.

The 1D scale vector is reshaped to match the position of the channel axis in the original tensor.

For a tensor of shape $(C,H,W)$, the scale is reshaped to $(C,1,1)$.

This enables element-wise operations where each tensor element is divided by the scale corresponding to its own channel, while maintaining high computational efficiency.

3. Symmetric Linear Quantization

The function linear_symmetric_quantize_per_channel performs the actual mapping from floating-point values to integers through the following steps:

Normalization:
Each floating-point value is divided by its channel-specific scale.

Discretization:
The normalized value is converted to an integer using rounding (or truncation, depending on the implementation).

Saturation (Clipping):
The result is clipped to the target integer range supported by the hardware, for example $[−127,127]$ for signed int8.

This ensures numerical stability and hardware compatibility.

4. Dequantization and Error Evaluation

Dequantization:
The inverse operation multiplies the quantized integer values by their corresponding channel scales to reconstruct an approximation of the original floating-point tensor. Since each channel uses its own scale, the reconstructed values closely match the original distribution.

Error metrics:

Mean Squared Error (MSE) quantifies the average magnitude of the quantization error.

Signal-to-Noise Ratio (SNR) measures the relative strength of the signal compared to quantization noise.

In practice, per-channel quantization typically achieves significantly higher SNR and lower reconstruction error than per-tensor quantization, particularly in models where channel-wise weight distributions vary substantially.

#### Asymmetric

```python
def linear_asymmetric_quantize_per_channel(
    tensor: np.ndarray, 
    scales: np.ndarray, 
    zero_points: np.ndarray, 
    axis: int = 0,
    dtype: np.dtype = np.int8,
    mode: str = 'round') -> np.ndarray:
    
    broadcast_shape = [1] * tensor.ndim
    broadcast_shape[axis] = -1
    s = scales.reshape(broadcast_shape)
    zp = zero_points.reshape(broadcast_shape)
    
    tensor_fp = tensor.astype(np.float32)
    
    if mode == 'round':
        q = np.round(tensor_fp / s + zp)
    elif mode == 'truncate':
        q = np.fix(tensor_fp / s + zp)
    
    qmin, qmax = np.iinfo(dtype).min, np.iinfo(dtype).max
    return np.clip(q, qmin, qmax).astype(dtype)

def linear_asymmetric_dequantize_per_channel(
    q_tensor: np.ndarray, 
    scales: np.ndarray, 
    zero_points: np.ndarray,
    axis: int = 0) -> np.ndarray:
    
    broadcast_shape = [1] * q_tensor.ndim
    broadcast_shape[axis] = -1
    s = scales.reshape(broadcast_shape)
    zp = zero_points.reshape(broadcast_shape)
    
    return (q_tensor.astype(np.float32) - zp) * s

def get_q_scale_and_zero_point_per_channel(
    tensor: np.ndarray, 
    axis: int = 0, 
    dtype: np.dtype = np.int8, 
    mode: str = 'round'):
    
    q_min = np.iinfo(dtype).min
    q_max = np.iinfo(dtype).max
    
    reduce_axes = tuple(i for i in range(tensor.ndim) if i != axis)
    r_min = tensor.min(axis=reduce_axes)
    r_max = tensor.max(axis=reduce_axes)
    
    scale = (r_max - r_min) / (q_max - q_min)
    scale = np.where(scale == 0, 1e-8, scale)
    
    zp = q_min - (r_min / scale)
    
    if mode == 'round':
        zp = np.round(zp)
    elif mode == 'truncate':
        zp = np.fix(zp)
    else:
        raise ValueError("mode should be 'round' or 'truncate'")
        
    zero_point = np.clip(zp, q_min, q_max).astype(dtype)
    
    return scale, zero_point
```

This implementation extends per-channel quantization by introducing **channel-specific zero-points** in addition to scales. Unlike symmetric quantization, asymmetric quantization allows the integer zero to map to a non-zero real value, making it better suited for tensors whose distributions are not centered around zero (e.g., activations).

1. **Per-Channel Scale and Zero-Point Computation**

`get_q_scale_and_zero_point_per_channel`

Instead of assuming symmetry around zero, this function computes **both scale and zero-point independently for each channel**.
- **Channel-wise reduction**:
    
    All tensor dimensions except the specified channel axis are reduced. This produces per-channel minimum (`r_min`) and maximum (`r_max`) values.
- **Scale computation**:

    The scale is computed as:
    $$s_c = \frac{r^{(c)}*{\max} - r^{(c)}*{\min}}{q_{\max} - q_{\min}}$$
    This maps the real-valued range of each channel to the full representable integer range.
- **Numerical stability**:
    
    Channels with zero dynamic range are handled by replacing zero scales with a small constant (`1e-8`) to avoid division by zero.

- **Zero-point calculation**:

    The zero-point shifts the quantized range so that the real minimum maps close to `q_min`:
    $$z_c = q_{\min} - \frac{r^{(c)}_{\min}}{s_c}$$

- **Discretization and clipping**:
  The zero-point is rounded or truncated (depending on `mode`) and clipped to the valid integer range before being cast to the target dtype.

This design allows each channel to fully utilize the available quantization range, even when the data is highly asymmetric.

2. Broadcasting Mechanism

Both quantization and dequantization rely on **NumPy broadcasting** to efficiently apply per-channel parameters:

- The scale and zero-point vectors are reshaped so that only the channel axis has size `C`, while all other dimensions are set to `1`.
- For example, a tensor with shape ((C, H, W)) uses scales and zero-points reshaped to ((C, 1, 1)).

This enables vectorized, element-wise operations without explicit loops, preserving performance and clarity.

3. **Per-Channel Asymmetric Quantization**

`linear_asymmetric_quantize_per_channel`

The quantization process follows the standard affine mapping:

- **Floating-point conversion**:
   The input tensor is cast to `float32` to ensure numerical precision.

- **Affine transformation**:
   Each value is normalized and shifted using its channel-specific parameters:
   $q = \frac{x}{s_c} + z_c$

- **Discretization**:
   The transformed value is rounded or truncated to an integer.

- **Saturation (Clipping)**:
   The result is clipped to the valid integer range defined by the target dtype (e.g., ([-128, 127]) for int8).

The output is a quantized tensor where each channel has been independently scaled and shifted.

4. **Per-Channel Dequantization**

`linear_asymmetric_dequantize_per_channel`

Dequantization reverses the affine mapping:

$$\hat{x} = (q - z_c) \cdot s_c$$

- The quantized tensor is converted back to `float32`.
- The channel-specific zero-point is subtracted.
- The result is multiplied by the channel-specific scale.

Because both scale and zero-point are applied per channel, the reconstructed tensor closely matches the original value distribution.

5. **Practical Implications**

- **Higher accuracy for non-zero-centered data**:

    Asymmetric quantization is particularly effective for activations and post-ReLU tensors.

- **Better range utilization**:

    Each channel maps its actual data range to the full integer range, reducing quantization error.

- **Trade-off**:

    The inclusion of zero-points increases computational complexity compared to symmetric quantization, which is why symmetric schemes are often preferred for weights, while asymmetric schemes are common for activations.

## Quantization Training Strategy
