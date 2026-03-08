import pandas as pd

def load_raw_data(path):
    df = pd.read_csv(path, index_col=0)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_data(df):
    df = df.drop(columns=[
        "mp_opp", "index_opp", "gmsc", "+/-",
        "gmsc_opp", "+/-_opp", "total", "total_opp"
    ])

    df.loc[df["ft%"].isna(), "ft%"] = 0
    df.loc[df["ft%_opp"].isna(), "ft%_opp"] = 0

    return df


def create_target(df):
    df = df.copy()
    df["target"] = df["won"].astype(int)
    return df

def compute_rolling_averages(df):
    cols_to_roll = ['fg', 'fga', 'fg%', '3p', '3pa', '3p%', 'ft', 'fta', 'ft%', 'orb',
       'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'ts%', 'efg%',
       '3par', 'ftr', 'orb%', 'drb%', 'trb%', 'ast%', 'stl%', 'blk%', 'tov%',
       'usg%', 'ortg', 'drtg', 'won', 'pts']
    
    df = df.sort_values("date").reset_index(drop=True)
    
    opp_cols_to_roll = [f"{col}_opp" for col in cols_to_roll]
    all_cols_to_roll = cols_to_roll + opp_cols_to_roll
    
    for col in all_cols_to_roll:
        if col in df.columns: # Safety check
            df[f"{col}_roll10"] = ( 
                df.groupby("team")[col]
                .rolling(10, closed='left')
                .mean()
                .reset_index(level=0, drop=True)
            )

  
    targets = ['won', 'pts', 'won_opp', 'pts_opp']
    cols_to_drop = [c for c in all_cols_to_roll if c not in targets]
    df = df.drop(columns=cols_to_drop)
    
    df = df.dropna() 

    rolling_stats_only = [col for col in df.columns if "_roll10" in col and "opp" not in col]
    lookup_df = df[['id', 'team'] + rolling_stats_only].copy()
    
    lookup_df = lookup_df.rename(columns={'team': 'team_opp'})
    for col in rolling_stats_only:
        lookup_df = lookup_df.rename(columns={col: f"{col}_opp_history"})
        
    df = df.merge(lookup_df, on=['id', 'team_opp'], how='left')
   
    df = df.dropna() 
    
    return df

def keep_home_games_only(df):
    df = df[df["home"] == 1].copy().reset_index(drop=True)
    df = df.drop(columns=["home","home_opp"])

    df = df.sort_values("date").reset_index(drop=True)
    return df

def calculate_elo(home_elo, away_elo, home_score, away_score, target):
    E_home = 1 / (1 + 10**((away_elo - home_elo) / 400))
    E_away = 1 / (1 + 10**((home_elo - away_elo) / 400))

    mov = abs(home_score - away_score)

    k = 20 * ((mov + 3)**0.8) / (7.5 + 0.006 * abs(home_elo - away_elo))

    new_home_elo = k * (target - E_home) + home_elo

    away_win = 1 - target  

    new_away_elo = k * (away_win - E_away) + away_elo
    return new_home_elo, new_away_elo

def compute_elo_feature(df):
    df = df.copy()

    df["home_elo"] = 1500
    df["away_elo"] = 1500
    df["home_elo"] = pd.Series(dtype='float64')
    df["away_elo"] = pd.Series(dtype='float64')


    teams = df["team"].unique()
    teams_dict = {} 
    year = df.iloc[0]["season"]

    for row in df.itertuples():

        if row.team_opp not in teams_dict:
            teams_dict[row.team_opp] = 1500

        if row.team not in teams_dict:
            teams_dict[row.team] = 1500

        if row.season > year:
            year = row.season
            for team in teams_dict:
                teams_dict[team] = teams_dict[team]*0.75 + 1505*0.25
        
        current_home_elo = teams_dict[row.team]
        current_away_elo = teams_dict[row.team_opp]
        
        df.at[row.Index, "home_elo"] = current_home_elo
        df.at[row.Index, "away_elo"] = current_away_elo
        
        new_home_elo, new_away_elo = calculate_elo(
            current_home_elo, current_away_elo, 
            row.pts, row.pts_opp, row.target
        )

        teams_dict[row.team] = new_home_elo
        teams_dict[row.team_opp] = new_away_elo
    
    return df

def build_features(path):
    if type(path)==str:
        df = load_raw_data(path)
    else:
        df = path

    df = clean_data(df)
    df = create_target(df)
    df = compute_rolling_averages(df)
    df = keep_home_games_only(df)
    df = compute_elo_feature(df)
    return df

