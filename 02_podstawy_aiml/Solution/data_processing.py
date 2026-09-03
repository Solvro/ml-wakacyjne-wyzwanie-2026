import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TitanicFeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        data = X.copy()
        self.age_by_group_ = data.groupby(["Pclass", "Sex"])["Age"].median()
        self.fare_by_group_ = data.groupby(["Pclass", "Sex"])["Fare"].median()
        self.age_median_ = data["Age"].median()
        self.fare_median_ = data["Fare"].median()
        self.embarked_mode_ = data["Embarked"].mode().iat[0]
        return self

    def transform(self, X):
        data = X.copy()
        groups = list(zip(data["Pclass"], data["Sex"]))
        age_fill_values = dict(zip(data.index, [self.age_by_group_.get(g, self.age_median_) for g in groups]))
        fare_fill_values = dict(zip(data.index, [self.fare_by_group_.get(g, self.fare_median_) for g in groups]))
        data["Age"] = data["Age"].fillna(age_fill_values)
        data["Fare"] = data["Fare"].fillna(fare_fill_values)
        data["Fare"] = np.log1p(data["Fare"])
        data["Embarked"] = data["Embarked"].fillna(self.embarked_mode_)
        data["Title"] = data["Name"].str.extract(r",\s*([^.]*)\.")[0]
        data["Title"] = data["Title"].where(data["Title"].isin(["Mr", "Miss", "Mrs", "Master"]), "Rare")
        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
        data["Deck"] = data["Cabin"].str[0].fillna("NoCabin")
        return data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilySize", "Deck"]]
