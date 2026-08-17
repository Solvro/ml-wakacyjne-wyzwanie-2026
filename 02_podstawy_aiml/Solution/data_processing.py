import pandas as pd
import numpy as np

def process_data(df):
    # categorical columns are already ok - no need for trimming

    # Passenger ID is already an index - no need to change it
    # Survived is already ok
    
    # Pclass is already ok

    # Name will be dropped, but after extracting the title, which is implemented in the feature engineering

    # Sex
    df["Sex"] = df["Sex"].map({'male': 0, 'female': 1}) # binary encoding

    # Age
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median())) # fill missing values with median age by Pclass and Sex

    # SibSp is already ok
    # Parch is already ok

    # Ticket - drop
    df.drop(columns=["Ticket"], inplace=True)

    # Fare
    df["Fare"] = df.groupby(["Pclass", "Sex"])["Fare"].transform(lambda x: x.fillna(x.median())) # fill missing values with median fare by Pclass and Sex
    df["Fare"] = np.log1p(df["Fare"]) # log-transform to reduce skewness

    # Cabin
    df["Cabin"] = df["Cabin"].fillna("NoCabin")
    # It will be dropped after extracting features, like Name

    # Embarked
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0]) # fill most frequent value
    df = pd.get_dummies(data=df, prefix='Embarked', columns=['Embarked'], dtype=np.int32) # one-hot encoding

    return df