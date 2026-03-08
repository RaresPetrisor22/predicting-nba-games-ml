from src.scraping.parser import make_games_csv
from src.scraping.scraper import get_games
from src.features.feature_engineer import build_features
from src.model.train import train_pipeline

def main():
    get_games()
    make_games_csv()
    df = build_features("nba_games.csv")
    train_pipeline(df)

if __name__ == "__main__":
    main()