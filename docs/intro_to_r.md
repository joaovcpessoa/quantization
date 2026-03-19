# Introdução a linguagem R

Frase de efeito não ajuda em nada pessoas ansiosas.

"R is a language for people who want to get things done. It's not a language for computer scientists; It's a language for scientists who compute."
- Hadley Wickham

Se tu gosta de assistir vídeos para aprender, vai minerar na internet para achar conteúdo que presta. A maioria dos cursos são superficiais e pouco interessantes.
A proposta desse documento não vai muito além disso. O diferencial seria o tom humorístico adicionado como tempero para quem acha que ler é chato e a exploração de alguns temas com maior profundidade que os vendedores de curso. 

## Instalação

Acesse https://cloud.r-project.org/ e baixe a versão mais recente do R para Windows, Mac ou Linux. Após baixar, realize o processo de instalação para habilitar a execução dos códigos. Se for utilizar o RStudio o problema é teu, vai ver tutorial na internet. Vou usar o VSCode, o que implica que ele utilizará o CMD para executar os códigos por baixo dos panos e você vai precisar adicionar o executável do R nas variáveis de ambiente, já que os desenvolvedores são preguiçosos e no processo de instalação não tem opção de fazer isso igual em quase qualquer linguagem existente na face da terra. Dito isso, considero que você foi capaz de fazer isso, afinal alguém que deseja utilizar uma linguagens para estatística avançada não saber pesquisar como instala algo no computador deveria gerar 20 pontos na carteira com bônus de suspensão da CNH.

## Sintaxe

Ao contrário de muitas outras linguagens de programação, em R você pode exibir código sem usar uma função de impressão. No entanto, existe uma função `print()` (Ninguém consegue se livrar dela) disponível para uso e há momentos em que você precisará usar ele para imprimir código, por exemplo, no cenário de loops usando `for`.

Obs.: EU NÃO GOSTO DE ASPA DUPLA! Pode me chamar de *tarado das aspas simples*, mas por algum motivo, olhar para um código com aspas duplas me gera um sentimento semelhante ao de entrar em uma cozinha suja e saber que você vai almoçar ali. Simplesmente eu acho desnecessário o símbolo, por isso você vai ver ele aparecendo aqui raras as vezes, quase como um *easter egg*.

Exemplo:
```R
print('Hello World!')
'Hello World!'
'100'
```

Os comentários servem tanto para explicar quanto impedir a execução de códigosm, como padrão entre as linguagens. Diferentemente de outras linguagens de programação, não existe sintaxe em R para comentários de múltiplas linhas. Vale ressaltar que se teu código tem muito comentário e ainda dá erro, provavelmente você é o problema.

Exemplo:
```R
# This is a comment
'Hello World!' # This is a comment
# 'Other code'
```

## Variáveis

Variáveis ​​são contêineres para armazenar valores de dados e em R não é diferente. As regras para nomes de variáveis ​​em R são:

1. O nome de uma variável deve começar com uma letra e pode ser uma combinação de letras, dígitos, ponto (.) e sublinhado (_). Se começar com um ponto (.), não pode ser seguido por um dígito
2. O nome de uma variável não pode começar com um número ou sublinhado (_)
3. Case Sensitive, ou seja, ​​diferenciam maiúsculas de minúsculas (age, Age e AGE são três variáveis ​​diferentes).
4. Palavras reservadas não podem ser usadas como variáveis ​​(TRUE, FALSE, NULL, if...)

Agora o que é importante. Uma variável é criada e declarada no momento em que você atribui um valor a ela pela primeira vez. Para fazer isso basta usar o sinal `<-` (De quem foi essa ideia ridícula, só aponta para eu matar o indivíduo). Para imprimir o valor da variável basta digitar seu nome. Em outras linguagens de programação, é comum usar `=` como operador de atribuição, em R é possível, porém em alguns contexto é proibido.

Exemplo:
```R
var1 <- 'Isso é ridículo'
var2 <- 7

var1
var2

# Nunca vi ninguém usar isso, mas vai que você é diferente
var3 <- var4 <- var5 <- 'Sou diferente'

var1 <- 'O valor da variável muda'
var1
```

É possível concatenar dois ou mais elementos usando a função `paste()`. Lembrando que se você tentar combinar uma *string* e um número, você obviamente vai gerar um erro (Isso aqui não é Javascript!).

Exemplo:
```R
text <- 'capaz de aprender'
paste('Não sou', text)

text1 <- 'Eu'
text2 <- 'desisto'
paste(text1, text2)
```

## Tipos de dados

Em R, as variáveis ​​não precisam ser declaradas com nenhum tipo específico e podem até mesmo mudar de tipo depois de terem sido definidas. Pela minha experiência de programação, isso é muito perigoso, mas quem sou eu.

Os tipos básicos de dados em R podem ser divididos nos seguintes tipos:

- `numeric`: 10.5, 55, 787
- `integer`: 1L, 55L, 100L (a letra 'L' indica que se trata de um número inteiro)
- `complex`: 9 + 3i ('i' é a parte imaginária)
- `character`: 'k', 'R é esquisito', 'FALSO', '11,5'
- `logical`: VERDADEIRO, FALSO

Para verificar o tipo de uma variável você pode usar a função `class()`.

Exemplo:
```R
# numeric
x <- 10.5
class(x)

# integer
x <- 1000L
class(x)

# complex
x <- 9i + 3
class(x)

# character/string
x <- 'R is exciting'
class(x)

# logical/boolean
x <- TRUE
class(x)
```

Você pode realizar conversão de tipo para os casos numérios com as funções.

- `as.numeric()`
- `as.integer()`
- `as.complex()`

Exemplo:
```R
x <- 1L
y <- 2

a <- as.numeric(x)
b <- as.integer(y)

x
y

class(a)
class(b)
```

Algumas curiosidades sobre o tipo de dado character. É possível atribuir uma string de várias linhas.

```R
str <- 'Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua'
str
cat(str)
nchar(str)
```

No entanto, observe que o R adicionará um `\n` ao final de cada quebra de linha. Isso é chamado de caractere de escape e o caractere *n* indica uma nova linha. Se você quiser que as quebras de linha sejam inseridas na mesma posição que no código, existe a função `cat()`, como eu mostrei.

Caso você queira calcular o número de caracteres em uma string, temos a função `nchar()`. O mais próximo que temos de regex em R é a função `grepl()`.

```R
str <- 'Hello World!'

grepl('H', str)
grepl('Hello', str)
grepl('X', str)
```

Para inserir caracteres inválidos em uma string, você deve usar um caractere de escape. No caso do R é a barra invertida `\` seguida do caractere que você deseja inserir.

```R
str <- 'We are the so-called \'Vikings\', from the north.'
str
```

Observe que a impressão automática da variável incluirá a barra invertida na saída. Para remover você pode usar a função `cat()` para imprimi-la sem a barra invertida (o que não faz o menor sentido).

Outros caracteres de escape em R:

| Code | Result          |
| ---- | --------------- |
| \\   | Backslash       |
| \n   | New Line        |
| \r   | Carriage Return |
| \t   | Tab             |
| \b   | Backspace       |

Para não deixar de falar dos valores lógicos, a comparação é realizada da seguinte forma.

```R
10 > 9    # TRUE
10 == 9   # FALSE
10 < 9    # FALSE
# Você também pode comparar duas variáveis:
a <- 10
b <- 9
a > b
```

## Operadores

O R divide os operadores nos seguintes grupos:

- Operadores aritméticos
- Operadores de atribuição
- Operadores de comparação
- Operadores lógicos
- Operadores diversos

**Operadores aritméticos**

Os operadores aritméticos são usados ​​com valores numéricos para realizar operações matemáticas comuns:

| Operador | Nome             | Exemplo |
| -------- | ---------------- | ------- |
| +	       | Adição           | x + y	|
| -	       | Subtração        | x - y   |
| *	       | Multiplicação	  | x * y	|
| /	       | Divisão          | x / y	|
| ^	       | Exponencial	  | x ^ y	|
| %%       | Módulo           | x %% y  |
| %/%      | Divisão Inteira  |	x%/%y   |

```R

```

**Operadores de atribuição**

Os operadores de atribuição são usados ​​para atribuir valores a variáveis:

```R
my_var <- 3
my_var <<- 3
3 -> my_var
3 ->> my_var
my_var
```

Nota: <<- é um atribuídor global. Também é possível inverter a direção do operador de atribuição. `x <- 3` é igual a `3 -> x`

**Operadores de comparação**

Os operadores de comparação são usados ​​para comparar dois valores:

| Operador | Nome                     |	Exemplo |
| -------- | ------------------------ | ------- |
| ==	   | Igualdade                | x == y  |
| !=	   | Diferença	              | x != y  |	
| >	       | Maior que  	          | x > y   |
| <	       | Menor que	              | x < y   |
| >=	   | Maior ou igual           |	x >= y  |	
| <=	   | Nenor ou igual	          | x <= y  |

Existem funções matemáticas integradas que permitem realizar operações matemáticas com números.

Exemplo:
```R
max(5, 10, 15) # retorna o valor máximo do conjunto
min(5, 10, 15) # retorna o valor mínimo do conjunto
sqrt(16)       # retorna a raíz quadrada
abs(-4.7)      # retorna o valor absoluto
ceiling(1.4)   # arredonda um número para cima, para o inteiro mais próximo
floor(1.4)     # arredonda um número para baixo, para o inteiro mais próximo
```

## Estruturas de dados

Para R temos as seguintes estruturas de dados:
- Vetores
- Listas
- Matrizes
- Arrays
- Dataframes

**Vetores**

Um vetor é a estrutura de dados mais básica em R. Ele contém uma lista de itens do mesmo tipo.

Exemplo:
```R
fruits <- c('banana', 'apple', 'orange')
numbers <- c(1, 2, 3)

fruits
numbers

# Vetor com valores numéricos em sequência
numbers <- 1:10
```

`c()` significa *combine*, junta os valores em um vetor

**Lists**

Uma lista pode armazenar diferentes tipos de dados em uma única estrutura. Você pode combinar números, strings, vetores e até mesmo outras listas.

Exemplo:
```R
thislist <- list('apple', 'banana', 50, 100)
thislist
```

**Matrices**

Uma matriz é uma estrutura de dados bidimensional onde todos os elementos são do mesmo tipo. É semelhante a uma tabela com linhas e colunas.

Exemplo:
```R
thismatrix <- matrix(c(1,2,3,4,5,6), nrow = 3, ncol = 2)
thismatrix
```

- `nrow` -> controla o tamanho de linhas
- `ncol` -> controla o tamanho das colunas

**Arrays**

Um *array* é semelhante a uma matriz, mas pode ter mais de duas dimensões. Ele armazena elementos do mesmo tipo em múltiplas dimensões.

Examplo:
```R
# array unidimensional com valores de 1 a 24
thisarray <- c(1:24)
# array multidimensional com valores de 1 a 24
multiarray <- array(thisarray, dim = c(4, 3, 2))

thisarray
multiarray

# Resultado
# [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
# , , 1 (Isso aqui indica a camada/matriz que está sendo mostrada)

#      [,1] [,2] [,3]
# [1,]    1    5    9
# [2,]    2    6   10
# [3,]    3    7   11
# [4,]    4    8   12

# , , 2

#      [,1] [,2] [,3]
# [1,]   13   17   21
# [2,]   14   18   22
# [3,]   15   19   23
# [4,]   16   20   24
```

A função `array()` cria um array multidimensional.
- `thisarray` -> os dados que serão usados (1 a 24)
- `dim = c(4,3,2)` -> define as dimensões do array, 4 linhas, 3 colunas e 2 matrizes

Os arrays são úteis para trabalhar com dados tridimensionais ou de dimensões superiores.

**Data Frames**

Um data frame é como uma tabela em uma planilha. Ele pode armazenar diferentes tipos de dados em várias colunas.

Examplo:
```R
Data_Frame <- data.frame (
  Training = c('Strength', 'Stamina', 'Other'),
  Pulse = c(100, 150, 120),
  Duration = c(60, 30, 45)
)

Data_Frame
```

---

## Estatística

Existe um conjunto de dados integrado popular no R chamado *mtcars* (Motor Trend Car Road Tests), que foi obtido da revista Motor Trend dos EUA de 1974. Nos exemplos abaixo, utilizaremos o esse conjunto para alguns aprendizados.

```R
mtcars

# Result:
#                      mpg cyl  disp  hp drat    wt  qsec vs am gear carb
# Mazda RX4           21.0   6 160.0 110 3.90 2.620 16.46  0  1    4    4
# Mazda RX4 Wag       21.0   6 160.0 110 3.90 2.875 17.02  0  1    4    4
# Datsun 710          22.8   4 108.0  93 3.85 2.320 18.61  1  1    4    1
# Hornet 4 Drive      21.4   6 258.0 110 3.08 3.215 19.44  1  0    3    1
# Hornet Sportabout   18.7   8 360.0 175 3.15 3.440 17.02  0  0    3    2
# Valiant             18.1   6 225.0 105 2.76 3.460 20.22  1  0    3    1
# Duster 360          14.3   8 360.0 245 3.21 3.570 15.84  0  0    3    4
# Merc 240D           24.4   4 146.7  62 3.69 3.190 20.00  1  0    4    2
# Merc 230            22.8   4 140.8  95 3.92 3.150 22.90  1  0    4    2
# Merc 280            19.2   6 167.6 123 3.92 3.440 18.30  1  0    4    4
# Merc 280C           17.8   6 167.6 123 3.92 3.440 18.90  1  0    4    4
# Merc 450SE          16.4   8 275.8 180 3.07 4.070 17.40  0  0    3    3
# Merc 450SL          17.3   8 275.8 180 3.07 3.730 17.60  0  0    3    3
# Merc 450SLC         15.2   8 275.8 180 3.07 3.780 18.00  0  0    3    3
# Cadillac Fleetwood  10.4   8 472.0 205 2.93 5.250 17.98  0  0    3    4
# Lincoln Continental 10.4   8 460.0 215 3.00 5.424 17.82  0  0    3    4
# Chrysler Imperial   14.7   8 440.0 230 3.23 5.345 17.42  0  0    3    4
# Fiat 128            32.4   4  78.7  66 4.08 2.200 19.47  1  1    4    1
# Honda Civic         30.4   4  75.7  52 4.93 1.615 18.52  1  1    4    2
# Toyota Corolla      33.9   4  71.1  65 4.22 1.835 19.90  1  1    4    1
# Toyota Corona       21.5   4 120.1  97 3.70 2.465 20.01  1  0    3    1
# Dodge Challenger    15.5   8 318.0 150 2.76 3.520 16.87  0  0    3    2
# AMC Javelin         15.2   8 304.0 150 3.15 3.435 17.30  0  0    3    2
# Camaro Z28          13.3   8 350.0 245 3.73 3.840 15.41  0  0    3    4
# Pontiac Firebird    19.2   8 400.0 175 3.08 3.845 17.05  0  0    3    2
# Fiat X1-9           27.3   4  79.0  66 4.08 1.935 18.90  1  1    4    1
# Porsche 914-2       26.0   4 120.3  91 4.43 2.140 16.70  0  1    5    2
# Lotus Europa        30.4   4  95.1 113 3.77 1.513 16.90  1  1    5    2
# Ford Pantera L      15.8   8 351.0 264 4.22 3.170 14.50  0  1    5    4
# Ferrari Dino        19.7   6 145.0 175 3.62 2.770 15.50  0  1    5    6
# Maserati Bora       15.0   8 301.0 335 3.54 3.570 14.60  0  1    5    8
# Volvo 142E          21.4   4 121.0 109 4.11 2.780 18.60  1  1    4    2
```

Vou comentar alguns comandos que podemos usar para verificar algumas informações.

Para obter algumas informações sobre os dados, use o ponto de interrogação:

```R
?mtcars
```

Para verificar a dimensão do conjuntos use a função `dim()`:

```R
Data_Cars <- mtcars
dim(Data_Cars)

# Result:
# [1] 32 11 
```

Para visualizar as variáveis das colunas use a função `names()`:

```R
Data_Cars <- mtcars
names(Data_Cars)

# Result: 
#  [1] "mpg" "cil" "disp" "hp" "drat" "wt" "qsec" "vs" "am" "marcha" 
# [11] "carb"
```

## Visualização de dados

