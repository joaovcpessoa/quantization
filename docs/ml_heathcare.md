### Age Assessment of Youth and Young Adults Using Magnetic Resonance Imaging of the Knee: A Deep Learning Approach

A tradução seria: Avaliação da Idade de Jovens e Adultos Jovens usando Ressonância Magnética do Joelho: Uma Abordagem de Deep Learning

O objetivo foi propor um método automatizado para estimar a idade cronológica de indivíduos entre 14 e 21 anos, focando na fase em que o crescimento cessa e as zonas de crescimento se fecham. O objetivo é evitar a radiação ionizante de métodos tradicionais baseados em raios-X.

Foram realizadas ressonâncias magnéticas (RM) do joelho de 402 voluntários (221 homens e 181 mulheres). O método empregou duas Redes Neurais Convolucionais (CNN), a primeira selecionava as imagens mais informativas da sequência de RM (focando na placa de crescimento, epífise e metáfise) e a segunda realizava a estimativa da idade. Diferentes arquiteturas foram testadas, sendo a GoogLeNet a que obteve melhores resultados.

O sistema conseguiu avaliar a idade de homens na faixa de 14 a 20,5 anos (erro médio de 0,793 anos) e de mulheres na faixa de 14 a 19,5 anos (erro médio de 0,988 anos). Na classificação de menores de 18 anos, a precisão foi de 98,1% para homens e 95% para mulheres. O método é capaz de realizar a avaliação de idade de forma totalmente automatizada e sem radiação, abordando as falhas dos métodos manuais tradicionais que sofrem com a variabilidade entre avaliadores.

### Avaliação da Idade Óssea com Diversas Técnicas de Machine Learning: Uma Revisão Sistemática da Literatura e Metanálise

*   **Objetivo:** Apresentar o estado da arte, tendências e lacunas na pesquisa sobre avaliação da idade óssea (AIO) que utilizam técnicas de **Aprendizado de Máquina (Machine Learning)**.
*   **Metodologia:** Foi realizada uma revisão sistemática da literatura (RSL) nas bases Pubmed, Scopus e Web of Science, selecionando **26 estudos** finais para análise de qualidade e extração de dados. Foi também realizada uma metanálise do desempenho dos sistemas propostos.
*   **Resultados:** A maioria dos estudos foca em radiografias de mão e punho. As técnicas mais comuns são regressão linear, Redes Neurais Artificiais e Máquinas de Vetores de Suporte (SVM). A metanálise indicou um erro médio absoluto (MAE) ponderado de **9,96 meses** na previsão da idade óssea.
*   **Conclusão:** Há uma clara dependência de radiografias, o que levanta questões éticas sobre exposição à radiação em jovens saudáveis. Existe uma lacuna em pesquisas que utilizem outras regiões (como o joelho) ou modalidades sem radiação (como RM), além de uma falta de exploração de fatores étnicos e socioeconômicos que influenciam o desenvolvimento ósseo.

### 3. Avaliação da Idade Cronológica em Indivíduos Jovens usando Estadiamento de Idade Óssea e Aspectos Não Radiológicos: Abordagem Multifatorial de Machine Learning (2020)

*   **Objetivo:** Investigar a estimativa da idade cronológica (IC) em indivíduos de 14 a 21 anos, combinando o estadiamento ósseo por RM com fatores não radiológicos (peso, altura, IMC e escala Tanner).
*   **Metodologia:** Foram analisadas imagens de RM de cinco zonas de crescimento (**rádio, tíbia distal, tíbia proximal, fêmur distal e calcâneo**) de **922 sujeitos** (455 homens e 467 mulheres). Radiologistas pediátricos avaliaram as imagens em 5 estágios de desenvolvimento. Diversos algoritmos de Machine Learning (como Random Forest e SVM) foram treinados para classificar os indivíduos como menores/adultos e para estimar a idade exata.
*   **Resultados:** A classificação de menores vs. adultos atingiu uma precisão de **90% para homens e 84% para mulheres**. No entanto, a estimativa da idade cronológica por grupos obteve erros médios de 0,95 e 1,24 anos, respectivamente, mostrando-se precisa apenas para as idades de 14 e 15 anos.
*   **Conclusão:** O estadiamento ósseo manual por RM, mesmo quando auxiliado por fatores antropométricos, não é preciso o suficiente para prever a idade cronológica exata em jovens no final da adolescência, devido à alta variação biológica (muitos já apresentam fusão completa aos 16 anos).

### 4. Técnicas de Machine Learning e Microssimulação no Prognóstico da Demência: Uma Revisão Sistemática da Literatura (2017)

*   **Objetivo:** Revisar como técnicas de Machine Learning (ML) e Microssimulação (MS) estão sendo aplicadas para o prognóstico da demência e suas comorbidades.
*   **Metodologia:** Revisão sistemática de **37 artigos** selecionados de bases de dados acadêmicas. O estudo analisou as técnicas utilizadas, os tipos de dados e os objetivos das previsões (individual ou populacional).
*   **Resultados:** O tema dominante (32 dos 37 estudos) é a previsão da conversão de Comprometimento Cognitivo Leve (MCI) para **Doença de Alzheimer**. A técnica de ML mais frequente é a SVM (30 estudos), e o dado mais utilizado é a **neuroimagem** (RM e PET), frequentemente da base ADNI. A Microssimulação foi usada em apenas dois estudos para estimar custos e o impacto de programas de triagem em nível populacional.
*   **Conclusão:** A pesquisa atual está muito focada em biomarcadores de imagem para prever a doença em curto prazo (até 36 meses). Há uma falta de abordagens holísticas que incluam fatores de estilo de vida, variáveis socioeconômicas e modelos epidemiológicos de longo prazo para populações.

### 5. Modelo de Previsão de Demência Multifatorial com 10 Anos de Antecedência (2020)

*   **Objetivo:** Desenvolver um modelo capaz de prever o diagnóstico de demência com **10 anos de antecedência**, utilizando uma ampla gama de fatores de risco modificáveis e não modificáveis.
*   **Metodologia:** Foram utilizados dados do estudo sueco SNAC, abrangendo **726 sujeitos** acompanhados por uma década. O modelo considerou **75 variáveis**, incluindo exames físicos (força manual, equilíbrio), testes bioquímicos, histórico médico, estilo de vida e instrumentos de saúde mental. Foi utilizada a técnica de **Árvore de Decisão (CART)** combinada com métodos para lidar com dados desbalanceados.
*   **Resultados:** O modelo alcançou uma área sob a curva (AUC) de **0,735** e uma sensibilidade (Recall) de **0,722** para prever a demência após 10 anos. A variável mais importante para a divisão inicial foi a **idade (limite de 75 anos)**. Outros fatores cruciais incluíram força física (teste de preensão manual), equilíbrio (ficar em um pé só), tabagismo, diabetes tipo 2 e IMC.
*   **Conclusão:** Fatores de risco modificáveis (como força física e hábitos de vida) são preditores mais eficazes a longo prazo do que os instrumentos de triagem cognitiva tradicionais (como o MEEM), que são mais voltados para o diagnóstico do que para o prognóstico precoce. O estudo sugere que intervenções nessas áreas poderiam, teoricamente, atrasar ou prevenir o início da demência.