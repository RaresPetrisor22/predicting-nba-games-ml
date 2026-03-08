from src.features.feature_engineer import build_features
from src.model.train import train_pipeline

def train_model():
    df = build_features("nba_games.csv")
    train_pipeline(df)

if __name__ == "__main__":
    train_model()