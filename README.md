# Technical Documentation

## Quantização

A quantização em ponto fixo requer um framework que suporte a conversão de tipos de dados sem perda severa de acurácia. Avaliei o PyTorch por sua flexibilidade em pesquisa, mas mantenho o TensorRT e o TFLite no radar para otimização específica de hardware. A performance final será determinada pelo backend de quantização escolhido.

O backend, nesse contexto, é o "motor" que realmente executa os cálculos matemáticos no hardware. 

Pensa no framework como o volante e no backend como o motor. Sem um backend específico, o processador tentaria rodar números inteiros (Int8) usando as mesmas instruções de ponto flutuante (Float32), o que não traria ganho nenhum de velocidade.

O backend serve para traduzir as operações da sua rede neural para instruções de baixo nível que o seu processador entende de forma otimizada. 

As principais funções são:

- Aceleração de Hardware: Ele utiliza instruções especiais do processador (como AVX-512 em Intel ou NEON em chips ARM) para processar vários dados de uma vez (SIMD).
- Gerenciamento de Memória: O backend organiza como os pesos da rede (agora menores, em Int8) são carregados no cache para evitar gargalos.- Aritmética de Precisão Reduzida: Ele garante que, ao multiplicar dois números de 8 bits, o resultado não "estoure" (overflow), gerenciando a re-quantização interna.2. 

O backend entra na fase de Inferência (quando o modelo já está treinado e pronto para uso). 

No PyTorch, o fluxo funciona assim:

- Modelo Original (Float32): Você define e treina a rede.
- Preparação/Mapeamento: Você escolhe o backend alvo.
- Conversão: O PyTorch ajusta os pesos para o formato que esse backend específico prefere.
- Execução: Quando você faz model(input), o PyTorch entrega os dados para o Backend, que faz o cálculo pesado e devolve o resultado.

Dependendo de onde você vai rodar sua rede, o backend muda:

| Hardware | Backend | Objetivo |
| -------- | --------- | ---------- |
| Servidores/PCs (Intel/AMD)         | fbgemm       | Otimizado para instruções x86 modernas | 
| Celulares/Raspberry Pi (ARM)       | qnnpack      | Focado em eficiência energética e chips mobile |
| Placas NVIDIA (Jetson/RTX)         | TensorRT     | É um backend e framework ao mesmo tempo, ultra-otimizado para GPUs |
| Microcontroladores (ESP32/Arduino) | TFLite Micro | Backend extremamente leve para rodar em poucos KB de RAM |

Se você configurar o backend errado (por exemplo, usar fbgemm para rodar em um Raspberry Pi), o código pode falhar ou até pior, funcionar, mas muito mais devagar do que a versão original sem quantização.

No nosso caso, como quero explorar a CPU, estou usando o `fbgemm`.

## Dataloader

Por hora não vou comentar sobre, mas basicamente ajusta os datasets para entrar no modelo corretamente e separa entre treino e validação.

## CNN without Observers

Essa CNN é bem simples e tem muito espaço para melhorias. O objetivo aqui é avaliar os efeitos da quantização então acho que ela será capaz de produzir resultados que colaborem nessa análise.

```python
class SimpleCNN(nn.Module):                    # Define class
    def __init__(self, num_classes):           # Construtor
        super().__init__()                     # Inicialize nn.Module
        
        self.conv = nn.Sequential(             # Creates the convolutional block.
            nn.Conv2d(3, 32, 3, padding=1),    # (3, 32, 32) -> (32, 32, 32)
            nn.ReLU(),                         # Add non-linearity
            nn.MaxPool2d(2),                   # Reduce dimension: 32x32 -> 16x16 (32, 16, 16)
 
            nn.Conv2d(32, 64, 3, padding=1),   # (32, 16, 16) -> (64, 16, 16)
            nn.ReLU(),                         # Add non-linearity
            nn.MaxPool2d(2)                    # Reduce dimension: 16x16 -> 8x8 (64, 8, 8)
        )

        self.fc = nn.Linear(64 * 8 * 8, num_classes) 

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
```

```python
class SimpleCNN(nn.Module):
```

Define uma nova classe de rede neural que herda de nn.Module, que é a classe base de todos os modelos no PyTorch.

```python
def __init__(self, num_classes):
```

É o construtor da classe. `num_classes` é o número de classes do seu problema.

```python
super().__init__()
```
Inicializa corretamente a classe base (nn.Module). Necessário para o PyTorch registrar parâmetros automaticamente.

```python
self.conv = nn.Sequential(
```
`nn.Sequential` cria um bloco onde as camadas são aplicadas em ordem automática. Evita ter que chamar cada camada manualmente no `forward`.

```python
nn.Conv2d(3, 32, 3, padding=1),
```

Significa:
- 3 → número de canais de entrada (RGB)
- 32 → número de filtros (feature maps de saída)
- 3 → kernel 3×3
- padding=1 → mantém tamanho espacial

Se entrada for (3, 32, 32) a saída será (32, 32, 32), ou seja, mesmo tamanho espacial, mas com 32 canais

```python
nn.ReLU(),
```

Função de ativação que introduz não-linearidade. Sem isso, a rede seria apenas uma transformação linear.

```python
nn.MaxPool2d(2),
```

Reduz dimensão pela metade.

Se entrada era: 32x32
Depois vira: 16x16

Isso reduz custo computacional, aumenta campo receptivo e ajuda generalização.

```python
nn.Conv2d(32, 64, 3, padding=1),
```

Segunda convolução. Agora:
- Entrada: 32 canais
- Saída: 64 canais
- Kernel 3×3
- Mantém tamanho espacial

Se entrada for (32, 16, 16), a saída será (64, 16, 16)

```python
nn.ReLU(),
```

Mesma função, adiciona não-linearidade.

```python
nn.MaxPool2d(2)
```

Reduz novamente pela metade: 16x16 → 8x8. Agora o tensor tem formato: (64, 8, 8)

Fecha o nn.Sequential.

```python
self.fc = nn.Linear(64 * 8 * 8, num_classes)
```

Parte Fully Connected. Após as convoluções, temos: 
- 64 canais
- 8 altura
- 8 largura

Total de features por imagem: 64 × 8 × 8 = 4096
A camada linear faz: 4096 → num_classes, ou seja, transforma features extraídas em pontuações (logits) para cada classe.

```python
def forward(self, x):
```

Define como os dados fluem pela rede.

```python
x = self.conv(x)
```

Aplica todo o bloco convolucional.

Se entrada era (batch, 3, 32, 32), agora vira (batch, 64, 8, 8)

```python
x = torch.flatten(x, 1)
```

Achata tudo menos o batch.

De: (batch, 64, 8, 8)
Para: (batch, 4096)

O 1 significa "Não mexa na dimensão 0 (batch), mas achate o resto"

```python
return self.fc(x)
```

Aplica a camada linear e retorna: (batch, num_classes). Isso são os logits (valores antes do softmax).

```python
cnn = CNN(num_classes=10).to(device)
```

Aqui é feito:
- Chamada do construtor da classe CNN
- Criação de um objeto do tipo CNN
- Inicializando:
    - As camadas convolucionais (self.conv)
    - A camada totalmente conectada (self.fc)

Esse objeto herda de `nn.Module`, então ele:
- Guarda os pesos da rede
- Guarda a arquitetura
- Tem métodos como .parameters(), .train(), .eval(), etc.

O método `.to(device)` move o modelo para CPU ou GPU, ou seja, movendo todos os pesos e buffers dessa rede para o dispositivo.

## 