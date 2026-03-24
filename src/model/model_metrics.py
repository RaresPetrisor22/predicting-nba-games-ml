import joblib
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve
from src.model.train import prepare_training_data
from src.features.feature_engineer import load_raw_data
import pandas as pd

def get_model_metrics():
    """
    Evaluates the model on the latest dataset and computes metrics.
    Paths are relative to the project root (where the app runs from).
    """
    pipeline = joblib.load("model_pipeline.pkl")
    df_eval = load_raw_data("data/nba_games_processed.csv").copy()
    
    if 'mp' not in df_eval.columns:
        df_eval['mp'] = 0

    df_test = df_eval[df_eval['season'] == 2026]
    X_test, y_test = prepare_training_data(df_test)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    y_proba_wins = y_prob[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    loss = float(log_loss(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    prob_true, prob_pred = calibration_curve(y_test, y_proba_wins, n_bins=10, strategy='uniform')
    
    return acc, loss, cm, prob_true.tolist(), prob_pred.tolist()

def load_and_process_time_series(csv_path="data/predictions.csv"):
    df = pd.read_csv(csv_path)
    
    df = df[df['actual'] != -1].copy()
    
    if df.empty:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    df['predicted_winner'] = (df['home_prob_win'] > 0.5).astype(int)
    
    dates = []
    cum_accuracies = []
    cum_log_losses = []
    
    unique_dates = df['date'].dt.date.unique()
    
    for date in unique_dates:
        history_df = df[df['date'].dt.date <= date]
        
        acc = accuracy_score(history_df['actual'], history_df['predicted_winner'])
        
        ll = log_loss(history_df['actual'], history_df['home_prob_win'], labels=[0, 1])
        
        dates.append(date)
        cum_accuracies.append(acc)
        cum_log_losses.append(ll)
        
    # Build a new dataframe specifically for plotting
    plot_df = pd.DataFrame({
        'Date': dates,
        'Cumulative Accuracy': cum_accuracies,
        'Cumulative Log Loss': cum_log_losses
    })
    
    return plot_df
    
