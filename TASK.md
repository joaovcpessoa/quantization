# TASKS

- [x] Definir um modelo de baseline.
- [x] É necessário quantizar o dataset?
- [x] Qual o efeito do truncameto e do arredondamento no processo de quantização?
- [ ] Construir um modelo quantizado 8-bits com saída antecipada
- [ ] Comparar as formas de reduzir o tamanho do modelo (redes neurais sem peso + redes esparsas)

## BASELINE

A princípio irei usar uma ResNet50 pré-treinada. Acho um modelo robusto e bem otimizado para problemas de visão computacional. A arquitetura dele se destaca por permitir o treinamento de redes muito profundas sem perda significativa de desempenho e considero isso ideal para comparar com um modelo quantizado. Nada impede de no futuro mudar.

## QUANTIZAR O DATASET?

Depende de quando e como estamos quantizando o modelo.

### Inferência com modelo quantizado

Se já temos um modelo quantizado (INT8) e vamos rodar inferência, não precisamos quantizat o dataset “offline”. O fluxo correto é:
- O dataset continua em float (FP32 / FP16)
- O modelo possui uma camada de quantização na entrada
- A ativação de entrada é quantizada **on-the-fly**, usando o `scale` e `zero_point` da primeira camada

Em frameworks como PyTorch/ONNX/TFLite, a entrada é FP32 e o `runtime` converte automaticamente para INT8 internamente, ou seja, você fornece dados normais e o modelo cuida do resto.

### Quantization Aware Training (QAT)

Se estamos fazendo QAT, o dataset continua em float e durante o forward pass o modelo **simula quantização** (fake quantization) e ativações e pesos sofrem arredondamento/truncamento *simulado*. Então você não quantiza o dataset previamente, já que a quantização acontece dentro do grafo do modelo.

### Post-Training Quantization (PTQ)

Durante a calibração você passa um dataset de calibração em float e o modelo usa esses dados para estimar `min/max` e calcular `scale` e `zero_point`. Ainda assim o dataset não é salvo quantizado. Ele é apenas usado para estimar estatísticas.

### A pergunta de 1 milhão de dólares

Quando faria sentido quantizar o dataset?

Hardware muito restrito
- Microcontroladores (MCUs)
- DSPs antigos
- FPGA sem unidade de float

Nesses casos o dataset pode ser armazenado já em INT8 para economizar memória e banda.
Mesmo assim, pelo que entendi, isso é uma decisão arquitetura, não de ML.

## EFEITO DO TRUNCAMENTO/ARREDONDAMENTO

```python
def linear_q_with_scale_and_zero_point(
    tensor, scale, zero_point, dtype = torch.int8):
    scaled_and_shifted_tensor = tensor / scale + zero_point
    rounded_tensor = torch.round(scaled_and_shifted_tensor) # AQUI É O PONTO CHAVE
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    q_tensor = rounded_tensor.clamp(q_min,q_max).to(dtype)   
    return q_tensor
```

- O método `torch.round` faz arredondamento para o inteiro mais próximo
Ex.: 2.3 → 2 / 2.7 → 3 / -1.5 → -2 (arredondamento “half away from zero” no PyTorch)

No caso do truncamento, seria simplesmente cortar a parte decimal: 
Ex.: 2.9 → 2 / -2.9 → -2

Pesquisei e vi que há um método para isso, o tal `torch.trunc()`. O fluxo dessa minha função é:

1. Escala e desloca: tensor / scale + zero_point
2. Arredonda para o inteiro mais próximo: torch.round(...)
3. Satura (clamp) no intervalo do tipo (`int8`, por padrão): [-128, 127]
4. Converte para inteiro (`torch.int8`)

Se quisermos realizar truncamento, seria algo como: 

```python
truncated_tensor = torch.trunc(scaled_and_shifted_tensor)
## Ou
truncated_tensor = scaled_and_shifted_tensor.to(torch.int32)
```

Observei o seguinte aqui. Considerando o valor real após escala:

$$ x = \frac{tensor}{scale} + zero_point $$

O erro de quantização $e = q - x$ fica em $e ∈ [−0.5, +0.5]$

Então nosso erro médio ≈ 0 e não há tendência sistemática para cima ou para baixo.

No truncamento o erro fica assim:

Para positivos: $e ∈ [−1,0]$
Para negativos: $e ∈ [0,1]$

- Erro médio ≠ 0 (viés)
- Valores positivos ficam sistematicamente subestimados
- Valores negativos ficam superestimados

Esse viés aparece mesmo que os dados sejam “bem distribuídos”.

Em uma rede neural:
- Uma camada já introduz viés
- Múltiplas camadas somam esse viés

Resultado: drift numérico

Isso pode causar:
- Mudança no offset das ativações
- Saturação mais frequente
- Perda de acurácia

Erro médio maior (MSE)

Arredondamento minimiza o erro quadrático médio (MSE).

Truncamento:
- Aumenta o erro médio
- Aumenta o erro absoluto médio
- Prejudica mais valores pequenos (próximos de zero)

Efeito específico em redes neurais
- Pesos
    - Truncamento tende a “puxar” pesos para zero
    - A rede fica menos expressiva
    - Pode parecer uma regularização, mas não controlada
- Ativações
    - Distribuições ficam deslocadas
    - BatchNorm / LayerNorm ficam inconsistentes
    - Saturação em int8 acontece mais cedo


