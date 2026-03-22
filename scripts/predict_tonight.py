import pandas as pd
import joblib
import os
from src.scraping.scraper import scrape_upcoming_games
from src.features.feature_engineer import build_features
from src.model.train import prepare_training_data
from src.scraping.scraper import ACTIVE_SEASON

def predict_tonight():
    print("Fetching tonight's matchups...")
    matchups = scrape_upcoming_games()
    
    if not matchups:
        print("No games scheduled for today. See you tomorrow!")
        return

    df = pd.read_csv("data/nba_games.csv",index_col=0)
    
    future_df = pd.DataFrame(matchups)
    
    future_df["season"] = ACTIVE_SEASON 
    for col in df.columns:
        if col not in future_df.columns:
            future_df[col] = 0
            
    future_df = future_df[df.columns]
            
    combined_df = pd.concat([df, future_df], ignore_index=True)
    combined_df = combined_df.sort_values("date").reset_index(drop=True)
    
    engineered_df = build_features(combined_df)
    
    num_games_tonight = len(matchups) // 2
    tonight_features_df = engineered_df.tail(num_games_tonight).copy()
    
    X_tonight, _ = prepare_training_data(tonight_features_df)
    
    pipeline = joblib.load("model_pipeline.pkl")
    
    probabilities = pipeline.predict_proba(X_tonight)

    preds_df = tonight_features_df[['id', 'date', 'team', 'team_opp']]
    preds_df['home_prob_win'] = probabilities[:,1]
    preds_df['away_prob_win'] = probabilities[:,0]
    preds_df['actual'] = -1


    if "predictions.csv" not in os.listdir("data"):
        preds_df.to_csv("data/predictions.csv", index=False)
    else:
        old_preds = pd.read_csv("data/predictions.csv")
        all_preds = pd.concat([old_preds, preds_df], ignore_index=True)
        if all_preds.duplicated().any():
            print("Games found already in predictions.csv. Skipping...")
            return
        preds_df.to_csv("data/predictions.csv", mode="a", header=False, index=False)

if __name__ == "__main__":
    predict_tonight()