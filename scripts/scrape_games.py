from src.scraping.parser import make_games_csv
from src.scraping.scraper import get_games
from src.features.feature_engineer import build_features
import os
import pandas as pd

def scrape_games():
    get_games()
    make_games_csv()
    df = build_features("data/nba_games.csv")
    df.to_csv("data/nba_games_processed.csv", index=False)
    
    if os.path.exists("data/predictions.csv"):
        preds_df = pd.read_csv("data/predictions.csv")
        scraped_df = df[['id','target']]
        scraped_df.rename(columns={'target':'actual'}, inplace=True)

        preds_df.set_index('id', inplace=True)
        scraped_df.set_index('id', inplace=True)
        
        preds_df.update(scraped_df[['actual']])
        preds_df.reset_index(inplace=True)

        preds_df.to_csv("data/predictions.csv", index=False)


if __name__ == "__main__":
    scrape_games()