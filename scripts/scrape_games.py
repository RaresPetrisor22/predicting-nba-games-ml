from src.scraping.parser import make_games_csv
from src.scraping.scraper import get_games
from src.features.feature_engineer import build_features

def scrape_games():
    get_games()
    make_games_csv()
    df = build_features("data/nba_games.csv")
    df.to_csv("data/nba_games_processed.csv", index=False)

if __name__ == "__main__":
    scrape_games()