import pandas as pd
import numpy as np

def engineer_features(df):
    # Title
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.") # extract title from name
    df["Title"] = df["Title"].where(
        df["Title"].isin(["Mr", "Miss", "Mrs", "Master"]), # keep common titles
        "Rare" # rest are mapped to 'Rare'
    )

    # drop Name
    df.drop(columns=["Name"], inplace=True)
    
    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # +1 to include the passenger themselves

    # HasCabin
    df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if x == "NoCabin" else 1) # 1 if has cabin

    # Deck
    df["Deck"] = df["Cabin"].str[0] # extract the letter standing for the deck
    df = pd.get_dummies(data=df, prefix="Deck", columns=["Deck"], dtype=np.int32)

    # drop Cabin
    df.drop(columns=["Cabin"], inplace=True)

    return df