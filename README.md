"""  **Desafios:**

1. Faça uma análise do dataset e nos explique quais as principais informações e como
elas se relacionam. Nesta análise deixe bem claro quais as suas expectativas e
quais as conclusões em cada passo da análise.
2. Enumere possíveis problemas que poderíamos resolver utilizando machine learning neste dataset. Construa um ou mais modelos e nos mostre suas habilidades demodelar um problema: decidindo features, labels e realizando uma experimentação.


Bônus*:
Dos modelos que você criou é possível fazer alguma análise de interpretação do modelo?
Tente encontrar qual o impacto das features no modelo, quais as mais importantes e como
elas podem ser interpretadas


# Importando as Bibliotecas necessárías

import pandas as pd
from matplotlib import pyplot
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn import metrics
import pydot
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# Análise de Dados - Conclusões

Ao realizar a Análise deste dataset, podemos observar que não há uma explicação clara das variáveis e portanto em diversas análises tive que hipotetizar o que entendia.
Então todas as linhas que apresentavam 0 = Não e 1 = Sim.

# Análise Descritiva

Para a realização desta análise, começei a observar a descrição análitica dos dados, desta formabpodemos observar que o nível de promessa do candidato de receber promoção dentre os 14999 apenas 21 tem chances, representando uma baixa porcentagem. 
Em relação a realocação, observamos que metade dos funcionários irão ser realocados, uma vez que visualizamos pela média.
Além disso, a grande maioria dos funcionários é importante para a organização, visto quea coluna de Trending Perf apresenta o 3º quartil, ou seja, 75% dos dados está com o valor de 3, com alta perfomance. 

A coluna Talent_Level apresenta todos os valores 6 o que indica uma coluna que não podemos diferenciaros funcionários, uma vez que é igual para todos. Além disso, nota-se que a empresa apresenta mais da metade dos funcionários trabalhando de forma remota, já que a coluna "Percent_Remote" ilustra na segunda linha a média de 0.61.

Verifica-se que a média da coluna "Linkedin_hits" apresenta uma diferença alta em relação a mediana, isso nos fornece informações que há outliers nesta colunas.

Observa-se que mais da metade das mulheres deixam a empresa, isso é evidente na coluna da média, ilustrando 70%, pode hipotetizar que não é uma empresa com uma cultura feminina ou que seja agradável para mulheres. Obviamente precisaria de mais dados para interpretar, mas podemos conjecturar.

Há 14% de chances de uma pessoa sofrer acidente no trabalho. Isso é verificado pelo terceiro quartil 
Verifica-se que poucas pessoas receberam promoções durante os últimos anos, uma vez que 75% está representado por 0.



* Após realizar uma análise descritiva, mensurei a correlação entre as variáveis:

Verificar a correlação entre os dados é importante, pois através da correlação podemos verificar se uma variável tende a aumentar ou diminuir, quando outra variável tende a aumentar ou diminuir em paralelo.

A correlação varia entre 1,0 e -1,0.

Coeficiente perfeitamente positivo:
r = 1,0
r> 0 à Uma relação positiva

Relação positiva significa que à medida que uma variável aumenta, a segunda variável também aumenta.
Coeficiente perfeitamente negativo:
r = -1,0
r <0 a uma relação negativa

Relação negativa significa que a medida que uma variável aumenta, a segunda variável segue na direção oposta.
r = 0 sem relacionamento.

Dessa forma, de acordo com o resultado acima, podemos verificar que ID e Percent_remote, apresenta a maior correlação positiva entre variaveis. Ou seja, a medida que aumenta a quantidade de id, vai aumentando o trabalho remoto, isso é notório, pois na análise descritiva acima, podemos perceber que há uma porcentagem maior de pessoas que trabalham remotamente.
Verifica-se também que há uma baixa correlação de dados entre ID e Critical.

Verifica-se, de forma geral, que há baixa correlação entre as variáveis. 

# Algumas conclusões foram:

* Verifica que ID e Percent_remote, apresenta a maior correlação positiva entre variaveis. Ou seja, a medida que aumenta a quantidade de id, vai aumentando o trabalho remoto, isso é notório, pois na análise descritiva acima, podemos perceber que há uma porcentagem maior de pessoas que trabalham remotamente.
Verifica-se também que há uma baixa correlação de dados entre ID e Critical.

* Pode-se concluir que as horas trabalhadas mensalmente e o resultado da avaliação no mês não melhoram o índice do funcionário na avaliação.


* Observa-se que a a correlação entre o índice obtido na ultima avaliação e o interesse em realocação é baixa e portanto, pode-se concluir que a ultima avaliação tem pouca influência na intenção em ser realocado. Além disso, a média de gasto de horas no mês e o tempo gasto na empresa apresentam uma baixa relação negativa, ou seja,a medida que aumenta o gasto de horas na empresa a disposição em ser realocado é menor.

* A Correlação entre tempo gasto na companhia e acidente no tabalho é baixa. Portanto, pode-se concluir que a o tempo gasto tem pouca influência em acidente de trabalho. Além disso, percebe-se que a média de gastos mensais com acidente de trabalho apresneta baixa correlação negativa, ou seja, a medida que aumenta o número de horas mensais a de ocorrência de acidente na empresa é menor.
Ademais, há uma baixa correlação entre ser realocado com o número de projetos, assim há baixa influência com o número de projetos em relação a realocação.

* Há uma baixa correlação entre o tempo gasto na companhia com o número de projetos, assim pode-se perceber que a quantidade de projetos que o funcionário assume, não ocasiona mudança no tempo de sua permanência na companhia.
A correlação entre o tempo gasto dentro de uma empresa e com a variável de promoção nos últimos 5 anos é baixa, dessa forma, nota-se que o tempo gasto dentro da empresa não está relacionado com a possibilidade de promoção na carreira do funcionário.
Observa-se também que, apesar de haver um baixo índice de correlação entre a promoção de um funcionário e a os funcionários que deixaram a empresa, este índice é negativo, implicando que a promoção na carreira pode potencialmente evitar que o funcionário deixe a empresa.

*A media de horas trabalhadas e conquistar a promoção nos ultimos 5 anos apresenta uma baixa relação negativa, ou seja,a medida que aumenta a promoção nos ultimos 5 anos na empresa a media de horas trabalhadas é menor.

*A ultima avaliação  e conquistar a promoção nos ultimos 5 anos apresenta uma baixa relação negativa, ou seja,a medida que aumenta a promoção nos ultimos 5 anos na empresa a ultima avaliação será menor.

*Além disso, ser realocado e conquistar a promoção nos ultimos 5 anos apresenta uma baixa relação negativa, ou seja,a medida que aumenta a promoção nos ultimos 5 anos na empresa e ser realocado é menor.

*Além disso, entre o número de projetos e conquistar a promoção nos ultimos 5 anos apresenta uma baixa relação negativa, ou seja,a medida que aumenta a promoção nos ultimos 5 anos na empresa a quantidade de números de projetos é menor.


# Expectativas

Os resultados foram bem diferentes do que havia previsto, geralmente o aumento do número de promoçoes influencia nas horas trabalhadas mensalmente pelo funcionário, uma vez que há mais responsabilidades.
Geralmente, ter números de projetos pode estar associado com conquistar a promoção, pode hipotetizar que é baixa, pois os projetos não derão certos ou não foram concluídos.
Além disso, se há mais tempo na empresa, pressupoe que terá uma média maior de horas mensais.
Ademais, conjecturo que a avaliação da perfomance interfere também em deixar a empresa, contudo, pelo cálculo, percebe-se que não há correlação

Verifico que uma grande parte do meu raciocínio foi diferente dos resultados, ao verificar no kaggle, vejo:

" "The fields represent a fictitious data set where a survey was taken and actual employee metrics exist for a particular organization. None of this data is real.""

Condiz não ser um dataset real

2.Enumere possíveis problemas que poderíamos resolver utilizando machine learning
neste dataset. Construa um ou mais modelos e nos mostre suas habilidades de
modelar um problema: decidindo features, labels e realizando uma experimentação

Bônus*:
Dos modelos que você criou é possível fazer alguma análise de interpretação do modelo?
Tente encontrar qual o impacto das features no modelo, quais as mais importantes e como
elas podem ser interpretadas.
*Esta é uma questão bônus e não é obrigatória

# Possíveis Modelagens:
 
Prever se ocorrerá acidente no trabalho - Árvore de Decisão
Prever a probabilidade de ocorrência de acidente no trabalho - Regressão Logística
Classificar os funcionários mais semelhantes - Clusterização K-MEANS
Prever se o funcionário terá uma alta perfomance - Árvore de Decisão
Prever se o número de projetos melhora a perfomance da pessoa - Árvore de Decisão
Prever se a quantidade de números de horas poderia deixar a companhia - Árvore de Decisão
A probabilidade de número de horas se ele deixaria a empresa - Regressão Logística
Determinar a probabibilidade de o tempo gasto influenciar em conquistar uma promoção nos últimos 5 anos - Árvore de Decisão


* Modelagens Construídas

# Regressão Logística
A Regressão Logística é uma técnica estatística que permite estimar a probabilidade associada à ocorrência de determinado evento. É um modelo que permite a predição de valores tomados por uma variável categórica, frequentemente binária.

Objetivo :Qual é a probabilidade de ocorrer um acidente de trabalho?
Raciocínio

O aumento da média de horas trabalhando em uma empresa pode afetar a produtividade, bem como pode comprometer com a qualidade do trabalho e com o cansaço, portanto, maiores deficits de atenção Em relação a variável a promotion_last_5years, pode conjecturar que obter uma promoção no trabalho, pode influenciar de certa forma na pressão de entregas de trabalho,similar ao mesmo pensamento foi construido com base no salário e na posição de trabalho

* Resultado

Obtive um resultado abaixo do esperado, embora a métrica do modelo foi alto. Hipótese é que há uma necessidade maior de pré-processamento dos dados e compreender melhor as variáveis disponíveis, contudo, não há uma explicação disponível. Além disso, conjectura que provavelmente é uma empresa que não tem indústria ou os funcionários inseridos não trabalham em áreas de indústrias ou em regiões com nivel de segurança máxima.Dessa forma, a probabilidade de ocorrência de acidente é baixa, tendo em vista que o modelo calcula valores iguais a zero para a totalidade dos dados analisados. Contudo, a hipótese que acredito é por que os dados estão desbalanceados, pode-se notar que há 1 - 14%,0 - 86%.


# Árvore de Decisão

A árvore de Decisão é um tipo de algoritmo de aprendizagem de máquina supervisionado que se baseia na ideia de divisão dos dados em grupos homogêneos

Objetivo: Classificar se os funcionários apresentam alta perfomance na empresa.
Raciocínio: Intuito de verificar se os funcionários apresentam alto perfomance. Dessa forma, parti da hipótese de que algumas variáveis poderiam influenciar na perfomance do funcionário com base na análise de dados realizada.Dessa forma, foi escolhida como variáveis independentes: 'number_project','average_montly_hours', 'Work_accident','promotion_last_5years','Role','salary' O Raciocínio foi de que a quantidade de números de projetos realizados pode interferir na perfomance do funcionário, uma vez que pode indicar que há alta produtividade. Em relação a "Work_accident" hipotetizou, pois caso tenha algum acidente no trabalho, pode diminuir a perfomance do funcionário. Enquanto que a promototion_last_5years pode ser um estimulo para o funcionário melhorar a perfomance. As variáveis "Role" e "salary" pode ser um fator pressionador para alto perfomance.

Resultado: A árvore não foi plotada, pois apresentou uma elevada quantidade de dados. Foi realizada algumas transformações de dados, retiradas de algumas colunas e diminuindo a porcentagem dos testes. Contudo, ainda assim permanceu uma árvore grande. Conjectura-se que havia uma necessidade de melhor processamento dos dados e melhor entendimento do dataset, contudo não há explicação no Kaggle a respeito deste dataset.





"""