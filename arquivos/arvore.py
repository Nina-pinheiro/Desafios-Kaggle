"""**Árvore de Decisão**

A árvore de Decisão é um tipo de algoritmo de aprendizagem de máquina supervisionado que se baseia na ideia de divisão dos dados em grupos homogêneos

* Objetivo: Classificar se os funcionários apresentam alta perfomance na empresa.

Raciocínio: Intuito de verificar se os funcionários apresentam alto perfomance.
Dessa forma, parti da hipótese de que algumas variáveis poderiam influenciar na perfomance do funcionário com base na análise de dados realizada.Dessa forma, foi escolhida como variáveis independentes: 'number_project','average_montly_hours', 'Work_accident','promotion_last_5years','Role','salary'
O Raciocínio foi de que a quantidade de números de projetos realizados pode interferir na perfomance do funcionário, uma vez que pode indicar que há alta produtividade. Em relação a "Work_accident" hipotetizou, pois caso tenha algum acidente no trabalho, pode diminuir a perfomance do funcionário. Enquanto que a promototion_last_5years pode ser um estimulo para o funcionário melhorar a perfomance. 
As variáveis "Role" e "salary" pode ser um fator pressionador para alto perfomance.

Resultado: A árvore não foi plotada, pois apresentou uma elevada quantidade de dados. Foi realizada algumas transformações de dados, retiradas de algumas colunas e diminuindo a porcentagem dos testes. Contudo, ainda assim permanceu uma árvore grande. Conjectura-se que havia uma necessidade de melhor processamento dos dados e melhor entendimento do dataset, contudo não há explicação no Kaggle a respeito deste dataset.
"""

# Separando o Dataset 70%

X_train_data = dados[['number_project','average_montly_hours', 
                      'Work_accident','promotion_last_5years','Role','salary']].iloc[0:9000,:] 
                      
Y_train_data = dados[['last_evaluation']].iloc[0:9000,:]


# Separando o Teste 30%

X_test_data = dados[['number_project','average_montly_hours', 'Work_accident',
                     'promotion_last_5years','Role','salary']].iloc[9001:14999,:]

Y_test_data = dados[['last_evaluation']].iloc[9001:14999,:]

# Treinando o modelo de arvore de decisão

clf = DecisionTreeRegressor()
clf = clf.fit(X_train_data,Y_train_data)

# Predição dos dados dos testes

resultado = clf.predict(X_test_data)

# Verificando os resíduos
clf.score(X_train_data,Y_train_data)


 # Visualização da árvore

tree = export_graphviz(clf, out_file ='tree8.dot', 
               feature_names =['number_project','average_montly_hours',
                               'Work_accident','promotion_last_5years','Role','salary'])
graph = graphviz.Source(tree)

# Visualização da árvore

tree.plot_tree (clf, feature_names=X_train_data.columns, class_names=Y_train_data, 
                label="all", filled=True, fontsize=6)

                