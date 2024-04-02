# https://www.kaggle.com/code/arezalo/customer-behaviour-prediction-naive-bayes#head2

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import warnings

from constants import *
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.naive_bayes import MultinomialNB
# from sklearn import metrics


if __name__ == '__main__':
    data = pd.read_csv(PATH)
    df = pd.DataFrame(data)

    print('\n\nOVERVIEW:\n\n')

    print(f"shape: {data.shape}")
    print('\nDataFrame Info:', df.info())

    # Check missing value
    print('\nMissing values:', df.isnull().sum().to_frame('NaN value').T)

    # check count of unique values in each column
    print('\nUnique Values:')
    for col in df:
        print(f"{col}: {df[col].nunique()}")

    # more details
    print('\nMore Details:')
    print(df.describe(include=[np.number]).T)
    print(df.describe(include=[object]).T)

    print("\n\nPREPARE:\n\n")

    print('\nBefore:', df)

    # Drop User ID columns
    df.drop('User ID', axis=1, inplace=True)

    # convert categoriacl feature to numerical:
    # only Gender is categorical
    df['Gender'] = df['Gender'].replace(['Male', 'Female'], [0, 1])

    print('\nAfter:', df)

    print("\n\nDATA ANALYSIS:\n\n")

    # check distribution of EstimatedSalary (based on Purchased)
    font = {'fontsize': 16, 'fontstyle': 'italic', 'backgroundcolor': 'black', 'color': 'orange'}
    plt.style.use('seaborn-notebook')
    sns.kdeplot(df.loc[df['Purchased'] == 0, 'EstimatedSalary'], label='No Purchased', shade=True)
    sns.kdeplot(df.loc[df['Purchased'] == 1, 'EstimatedSalary'], label='Purchased', shade=True)
    plt.title('KDE of EstimatedSalary (based on Purchased)', fontdict=font, pad=15)
    plt.xticks(np.arange(0, 200001, 10000), rotation=90)
    plt.xlim([0, 200001])
    plt.legend()
    plt.show()

    # check distribution of Purchased (based on Purchased)
    plt.style.use('seaborn-notebook')
    sns.kdeplot(df.loc[df['Purchased'] == 0, 'Age'], label='No Purchased', shade=True)
    sns.kdeplot(df.loc[df['Purchased'] == 1, 'Age'], label='Purchased', shade=True)
    plt.title('KDE of Age (based on Purchased)', fontdict=font, pad=15)
    plt.xticks(np.arange(0, 70, 5))
    plt.xlim([10, 70])
    plt.legend()
    plt.show()
