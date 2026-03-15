import pandas as pd
import joblib
from datetime import datetime
from src.scraping.scraper import scrape_upcoming_games
from src.features.feature_engineer import build_features
from src.model.train import prepare_training_data
from src.scraping.scraper import ACTIVE_SEASON

def display_results(results_df,probabilities):
    results = []
    for i, (_,row) in enumerate(results_df.iterrows()):
        home_team = row["team"]
        away_team = row["team_opp"]
        
        # Index 1 is the probability of the '1' class (which means the Home team wins)
        home_win_prob = probabilities[i][1] * 100
        away_win_prob = probabilities[i][0] * 100
        
        results.append({
            "home": home_team,
            "away": away_team,
            "home_prob": round(home_win_prob, 1),
            "away_prob": round(away_win_prob, 1)
        })
    return results

def predict_tonight():
    print("Fetching tonight's matchups...")
    matchups = scrape_upcoming_games()
    
    if not matchups:
        print("No games scheduled for today. See you tomorrow!")
        return []

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
    
    print("\n" + "="*40)
    print(f"🏀 PREDICTIONS FOR {datetime.now().strftime('%Y-%m-%d')} 🏀")
    print("="*40)

    probabilities = pipeline.predict_proba(X_tonight)
   
    return display_results(tonight_features_df,probabilities)

if __name__ == "__main__":
    preds = predict_tonight()
    for p in preds:
        print(f"{p['home']} (Home) vs {p['away']} (Away)")
        print(f"  -> {p['home']} Win Probability: {p['home_prob']:.1f}%")
        print(f"  -> {p['away']} Win Probability: {p['away_prob']:.1f}%")
        print("-" * 40)