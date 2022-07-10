import time
import numpy as np
import pandas as pd
import streamlit as st
import sklearn.metrics as metrics
import sklearn.datasets as datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#### Barra Superior da página ####

# Título
st.write('*UI para Experimentos de Machine Learning*')
st.write('*Aplicando modelos para os Datasets do Scikit-Learn*')
st.title('Regressão Logística')


#### Barra lateral de Navegacao ####

st.sidebar.header('Dataset e Hiperparâmetros')               # Abre uma barra lateral de navegação com cabeçalho
st.sidebar.markdown("**Selecione o Dataset Desejado**")      # Camada de texto no sidebar
Dataset = st.sidebar.selectbox('Dataset',('Iris','Wine','Breast Cancer'))  # Add Selectbox no sidebar
Split = st.sidebar.slider('Escolha o percentual de separação treino/teste',0.1,0.9,0.7)

st.sidebar.markdown('**Selecione os Hiperparâmetros da Regressão Logistica**')
Solver = st.sidebar.selectbox('algoritmo', ('lbfgs', 'newton-cg', 'liblinear', 'sag'))
Penality = st.sidebar.radio('Regularização', ('none', 'l1', 'l2', 'elasticnet'))
Tol = st.sidebar.text_input('Tolerancia Para Critério de Parada:',"1e-4")
Max_Iteration = st.sidebar.text_input('Número de Iterações:', '50')
Scaler = st.sidebar.radio('Estratégia de Normalizacao', ('MinMaxScaler', 'StandardScaler'))

# Dicionário de Hiperparametros de Treinamento da RegLog

parameters = {'Penality':Penality,
              'Tol':Tol,
              'Max_Iteration':Max_Iteration,
              'Solver':Solver}


#### Carga do Dataset e Preprocessamento ####

# Funcao para Carga do Dataset

def carrega_dataset(dataset):
    if dataset == 'Iris':
        dados = datasets.load_iris()
    elif dataset == 'Wine':
        dados = datasets.load_wine()
    elif dataset == 'Breast Cancer':
        dados = datasets.load_breast_cancer()

    return dados


def prepara_dados(dados, split, scal):

    #Dividir o dataset em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(dados.data, dados.target, train_size=float(split))

    # Normalizando os dados de treino
    if scal == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scal == 'StandardScaler':
        scaler = StandardScaler()

    # Fit e transform nos dados de treino
    X_train = scaler.fit_transform(X_train)

    # Apenas Transform nos Dados de Teste
    X_test = scaler.transform(X_test)

    return (X_train, X_test, y_train, y_test)


### Build do Modelo ###

def criar_modelo(parameters):

    # Extrair Dados Preparados para Treino e Teste do Modelo
    X_train, X_test, y_train, y_test = prepara_dados(Data, Split, Scaler)

    # Instanciando o modelo e treinando o modelo
    clf = LogisticRegression(penalty=parameters['Penality'],
                             solver=parameters['Solver'],
                             max_iter=int(parameters['Max_Iteration']),
                             tol=float(parameters['Tol'])).fit(X_train,y_train)
    # Predições
    y_pred = clf.predict(X_test)

    # Avaliação da Acuracia
    accuracy = metrics.accuracy_score(y_test,y_pred)

    # Matriz de Confusão
    cm = metrics.confusion_matrix(y_test,y_pred)

    # Dicionário de Resultados
    dict_value = {'modelo':clf,
                  'acuracia':accuracy,
                  'previsao':y_pred,
                  'y_real':y_test,
                  'Metricas':cm,
                  'X_teste':X_test}

    return (dict_value)
    #return X_train, X_test, y_train, y_test

##### Desenvolvimento Corpo da Página #####

# Resumo dos Dados

st.markdown("""Resumo dos Dados""")
st.write('Nome do Dataset: ',Dataset)

# Carrega o Dataset escolhido
Data = carrega_dataset(Dataset)

# Extraindo as targets
targets = Data.target_names

# Prepara o Dataframe com os Dados

Dataframe = pd.DataFrame(Data.data, columns=Data.feature_names)
Dataframe['target'] = pd.Series(Data.target)
Dataframe['target labels'] = pd.Series(targets[i] for i in Data.target)

# Mostrando dataset selecionado
st.write('Visão Geral dos Dados Selecionados')
st.write(Dataframe)


if(st.sidebar.button('Treinar Modelo')):
    # Barra de Progressão
    with st.spinner('Carregando o Dataset...'):
        time.sleep(0.1)

    # Info sucesso
    st.success('Dataset Carregado')

    # trainando o modelo
    modelo = criar_modelo(parameters)

    # Barra de Progressão do Treinamento do Modelo
    my_bar = st.progress(0)

    # Mostar barra de Progressão
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete+1)

    # Info usuário
    with st.spinner('Treinando...'):
        time.sleep(1)

    # Info de sucesso
    st.success("Modelo Treinado")

    # Extraindo labels
    labels_reais = [targets[i] for i in modelo['y_real']]
    # Extraindo Previsoes
    labels_previstas = [targets[i] for i in modelo['previsao']]

    # Sub titulo
    st.subheader("Previsões do modelo nos Dados de Teste")

    # Mostrar Resultados
    st.write(pd.DataFrame({'Valor Real':modelo['y_real'],
                           "Label Real":labels_reais,
                           'Valor Previsto':modelo['previsao'],
                           'Label Prevista':labels_previstas}))
    # Matriz de Confusão
    matriz = modelo['Metricas']
    st.subheader('Matriz de Confusão nos Dados de Teste')
    st.write(matriz)

    # Acuracia
    st.write('A Acurária do modelo é: ', modelo['acuracia'])

    # fim.
    st.write('Fim do processo. Até a próxima ... ')











