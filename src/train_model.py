"""
Esse é um script simples que foi criado para fazermos o treinamento do modelo utilizando o conjunto de dados do Titanic
e o pipeline da Feature Engineering que mostramos no Workshop. Para executá-lo, você deve estar na raiz do projeto e digitar
no seu terminar o seguinte comando:

python src/train_model.py

Ele irá receber o conjunto de treinamento, rodar o pipeline da feature engineering com ele e depois fazer o treinamento
do modelo utilizando o conjunto de dados de treinamento transformado pelo pipeline da feature engineering.

Após isso, também temos uma função que irá fazer predições para os conjuntos de treino e teste e mostrar as métricas para
o conjunto de treinamento (conjunto de teste não tem o target, então só conseguimos ver a performance do modelo no site do
Kaggle como mostramos no Workshop).

Por fim, salvamos os conjuntos de dados de treino e teste com as predições e também o modelo treinado no formato pickle.

"""

import os
import sys
sys.path.insert(1, os.getcwd())

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from feature_engineering.feature_engineering_pipeline import FeatureEngineering
from sklearn.metrics import accuracy_score
from src.utils import save_pickle


def transform_feature_engineering(X_train, X_test):
    # instancia o pipeline da feature engineering
    feature_engineering_pipeline = FeatureEngineering(numerical_features=['Pclass', 'Age']).get_pipeline()

    # "aprende" e faz as transformações da feature engineering no conjunto de treinamento
    index = X_train.index
    X_train = feature_engineering_pipeline.fit_transform(X_train)
    X_train['PassengerId'] = index
    X_train = X_train.set_index('PassengerId')

    # faz as transformações da feature engineering no conjunto de teste
    index = X_test.index
    X_test = feature_engineering_pipeline.transform(X_test)
    X_test['PassengerId'] = index
    X_test = X_test.set_index('PassengerId')

    # salva os conjuntos de dados transformados pela feature engineering
    X_train.to_csv('data/train_after_feature_engineering.csv')
    X_test.to_csv('data/test_after_feature_engineering.csv')

    return X_train, X_test


def generate_predictions(model, df):
    # gera as predições para o conjunto de dados fornecido com o modelo treinado
    df['prediction'] = model.predict_proba(df)[:, 1]
    df['Survived'] = model.predict(df.drop('prediction', axis=1))
    return df


def main():
    # importa o conjunto de treinamento e salva as features e o target em variáveis distintas
    train = pd.read_csv('data/train.csv', index_col='PassengerId')
    X_train = train.drop(['Survived'], axis=1)
    y_train = train['Survived']

    # importa o conjunto de teste
    X_test = pd.read_csv('data/test.csv', index_col='PassengerId')

    # seleciona somente as features que serão fornecidas para o modelo
    features = ['Pclass', 'Age', 'Sex']
    X_train = X_train[features]
    X_test = X_test[features]

    # faz as transformações da feature engineering nos conjuntos de treino e teste
    X_train, X_test = transform_feature_engineering(X_train, X_test)

    # treina o modelo com o conjunto de treinamento
    model = LogisticRegression(verbose=1, max_iter=1000)
    model.fit(X_train, y_train)

    # gera as predições e calcula a acurácia para o conjunto de treinamento
    df = generate_predictions(model, X_train)
    df.to_csv('data/train_predictions.csv')
    print(f'Acurácia do conjunto de treinamento: {accuracy_score(y_train, df["Survived"])}')

    # gera as predições para o conjunto de teste
    df = generate_predictions(model, X_test)
    df.to_csv('data/test_predictions.csv')

    # salva o modelo treinado
    save_pickle(model, 'models/model.pkl')


if __name__ == '__main__':
    main()
