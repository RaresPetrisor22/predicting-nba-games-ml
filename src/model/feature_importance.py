import joblib
import pandas as pd
import matplotlib.pyplot as plt
from src.model.train import prepare_training_data
from src.features.feature_engineer import build_features

def get_feature_importance():
    pipeline = joblib.load("model_pipeline.pkl")

    lr_model = pipeline.named_steps['logisticregression']

    weights = lr_model.coef_[0]

    df = build_features('data/nba_games.csv')
    X, _ = prepare_training_data(df)
    features = list(X.columns) 

    importance_df = pd.DataFrame({
        'Feature': features,
        'Weight': weights
    })

    importance_df['Abs_Weight'] = importance_df['Weight'].abs()

    top_features = importance_df.sort_values(by='Abs_Weight', ascending=False).head(15)

    return top_features



