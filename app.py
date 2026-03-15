import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Import custom feature engineering functions
from src.features.feature_engineer import clean_data, create_target, compute_rolling_averages, compute_elo_feature
from scripts.predict_tonight import predict_tonight

# --- TEAM DICTIONARY ---
NBA_TEAMS = {
    'ATL': {'name': 'Atlanta Hawks', 'id': '1610612737'},
    'BOS': {'name': 'Boston Celtics', 'id': '1610612738'},
    'BRK': {'name': 'Brooklyn Nets', 'id': '1610612751'},
    'CHI': {'name': 'Chicago Bulls', 'id': '1610612741'},
    'CHO': {'name': 'Charlotte Hornets', 'id': '1610612766'},
    'CLE': {'name': 'Cleveland Cavaliers', 'id': '1610612739'},
    'DAL': {'name': 'Dallas Mavericks', 'id': '1610612742'},
    'DEN': {'name': 'Denver Nuggets', 'id': '1610612743'},
    'DET': {'name': 'Detroit Pistons', 'id': '1610612765'},
    'GSW': {'name': 'Golden State Warriors', 'id': '1610612744'},
    'HOU': {'name': 'Houston Rockets', 'id': '1610612745'},
    'IND': {'name': 'Indiana Pacers', 'id': '1610612754'},
    'LAC': {'name': 'Los Angeles Clippers', 'id': '1610612746'},
    'LAL': {'name': 'Los Angeles Lakers', 'id': '1610612747'},
    'MEM': {'name': 'Memphis Grizzlies', 'id': '1610612763'},
    'MIA': {'name': 'Miami Heat', 'id': '1610612748'},
    'MIL': {'name': 'Milwaukee Bucks', 'id': '1610612749'},
    'MIN': {'name': 'Minnesota Timberwolves', 'id': '1610612750'},
    'NOP': {'name': 'New Orleans Pelicans', 'id': '1610612740'},
    'NYK': {'name': 'New York Knicks', 'id': '1610612752'},
    'OKC': {'name': 'Oklahoma City Thunder', 'id': '1610612760'},
    'ORL': {'name': 'Orlando Magic', 'id': '1610612753'},
    'PHI': {'name': 'Philadelphia 76ers', 'id': '1610612755'},
    'PHO': {'name': 'Phoenix Suns', 'id': '1610612756'},
    'POR': {'name': 'Portland Trail Blazers', 'id': '1610612757'},
    'SAC': {'name': 'Sacramento Kings', 'id': '1610612758'},
    'SAS': {'name': 'San Antonio Spurs', 'id': '1610612759'},
    'TOR': {'name': 'Toronto Raptors', 'id': '1610612761'},
    'UTA': {'name': 'Utah Jazz', 'id': '1610612762'},
    'WAS': {'name': 'Washington Wizards', 'id': '1610612764'}
}

def get_logo_url(team_id):
    if not team_id:
        return "https://cdn.nba.com/logos/nba/global/primary/L/logo.svg"
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NBA Predictor Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR NBA THEME ---
st.markdown("""
    <style>
    /* Main body background */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #1D428A; /* NBA Blue */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #1D428A;
        border: 1px solid #dee2e6;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1D428A;
        color: white !important;
        border-top: 3px solid #C8102E; /* NBA Red */
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #C8102E; /* NBA Red */
        font-size: 2rem;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #1D428A; /* NBA Blue */
        font-size: 1.1rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-top: 4px solid #1D428A;
        margin-bottom: 15px;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #1D428A;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_and_process_data():
    # Load raw data
    df = pd.read_csv("data/nba_games.csv", index_col=0)
    df = df.sort_values("date").reset_index(drop=True)
    
    # We want to use the user's functions, but we need the data BEFORE `keep_home_games_only`
    # so we can track a team whether they are home or away easily for analytics/elo.
    # So we will replicate `build_features` partially to get Elo and rolling stats
    
    df_clean = clean_data(df)
    df_target = create_target(df_clean)
    
    # Calculate Rolling Averages
    df_rolling = compute_rolling_averages(df_target)
    
    # Calculate Elo (requires `home_elo` and `away_elo` computation)
    # We can pass df_rolling to compute_elo_feature
    df_elo = compute_elo_feature(df_rolling)
    
    # For easier querying by team, we'll create a unified view of team performance.
    return df_elo



# --- MAIN APP ---
def main():
    st.title("🏀 NBA Predictor Dashboard")
    st.markdown("An interactive dashboard for the NBA Machine Learning predictive pipeline.")
    
    try:
        df = load_and_process_data()
        
        # Get list of all unique teams
        all_teams = sorted(list(df['team'].unique()))
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Create Tabs
    tab1, tab2, tab3 = st.tabs([
        "📅 Tonight's Predictions", 
        "📊 Team Analytics (Last 10 Games)", 
        "📈 Historical Elo Tracker"
    ])

    # --- TAB 1: Tonight's Predictions ---
    with tab1:
        st.header("Tonight's Matchups")
        st.markdown(f"**Date:** {datetime.now().strftime('%B %d, %Y')}")
        
        with st.spinner("Fetching tonight's matchups and running predictions..."):
            matchups = predict_tonight()
        
        if not matchups:
            st.info("No games scheduled for today. See you tomorrow!")
        else:
            for match in matchups:
                home_abbr = match['home']
                away_abbr = match['away']
                home_info = NBA_TEAMS.get(home_abbr, {'name': home_abbr, 'id': ''})
                away_info = NBA_TEAMS.get(away_abbr, {'name': away_abbr, 'id': ''})
                
                # Use dedented HTML to avoid Markdown treating it as a code block (due to leading spaces)
                match_html = f"""
<div class="metric-container" style="display: flex; justify-content: space-between; align-items: center; padding: 20px;">
<!-- Home Team -->
    <div style="display: flex; align-items: center; width: 40%;">
        <img src="{get_logo_url(home_info['id'])}" style="height: 70px; margin-right: 15px;">
        <div>
            <div style="color: #1D428A; font-size: 1.2rem; font-weight: bold;">{home_info['name']}</div>
            <div style="color: #C8102E; font-size: 2.2rem; font-weight: bold;">{match['home_prob']}%</div>
        </div>
    </div>
    
<!-- VS -->
<div style="width: 20%; text-align: center;">
        <h3 style="color: #1D428A; margin: 0; font-size: 1.5rem;">VS</h3>
</div>
    
<!-- Away Team -->
<div style="display: flex; align-items: center; justify-content: flex-end; width: 40%; text-align: right;">
    <div>
        <div style="color: #1D428A; font-size: 1.2rem; font-weight: bold;">{away_info['name']}</div>
        <div style="color: #C8102E; font-size: 2.2rem; font-weight: bold;">{match['away_prob']}%</div>
    </div>
    <img src="{get_logo_url(away_info['id'])}" style="height: 70px; margin-left: 15px;">
</div>
</div>
"""
                st.markdown(match_html, unsafe_allow_html=True)

    # --- TAB 2: Team Analytics ---
    with tab2:
        st.header("Team Analytics")
        
        selected_team = st.selectbox("Select a Team:", all_teams, key="team_select_analytics")
        
        # Filter games where the selected team is playing (either home or away)
        # Note: The dataframe from `compute_rolling_averages` dropped `total`, etc.
        # But 'team' is always the primary team for that row in the original df.
        team_games = df[df['team'] == selected_team].copy()
        team_games = team_games.sort_values(by='date', ascending=False)
        
        last_10 = team_games.head(10)
        
        st.subheader(f"Last 10 Games Overview for {selected_team}")
        
        # Find rolling average columns
        roll_cols = [c for c in df.columns if "roll10" in c and "opp" not in c]
        
        # Basic box score columns (+ date, opponent)
        display_cols = ['date', 'team_opp', 'won', 'pts', 'pts_opp'] + roll_cols
        
        # Keep only columns that exist
        display_cols = [c for c in display_cols if c in last_10.columns]
        
        st.dataframe(
            last_10[display_cols].style.format(precision=2),
            use_container_width=True,
            hide_index=True
        )

    # --- TAB 3: Historical Elo Tracker ---
    with tab3:
        st.header("Historical Elo Tracker")
        
        selected_team_elo = st.selectbox("Select a Team:", all_teams, key="team_select_elo")
        
        # We need to extract the Elo for the selected team over time
        # The team could be in 'team' or 'team_opp'
        # To get a chronological series of the team's Elo:
        
        # Filter rows where the team played
        team_filter = (df['team'] == selected_team_elo) | (df['team_opp'] == selected_team_elo)
        team_elo_df = df[team_filter].copy()
        
        # Extract the correct Elo depending on if they were 'team' or 'team_opp'
        # Assuming `home_elo` corresponds to `team` if home==1, wait: `home_elo` and `away_elo` 
        # were created based on `team` and `team_opp` in `compute_elo_feature`.
        # Looking at `feature_engineer.py`: 
        # current_home_elo = teams_dict[row.team]
        # current_away_elo = teams_dict[row.team_opp]
        # So `home_elo` is just the elo of `team`, and `away_elo` is the elo of `team_opp`.
        
        def get_team_elo(row):
            if row['team'] == selected_team_elo:
                return row['home_elo']
            else:
                return row['away_elo']
                
        team_elo_df['Team_Elo'] = team_elo_df.apply(get_team_elo, axis=1)
        team_elo_df = team_elo_df.sort_values('date')
        
        # Plotting
        fig = px.line(
            team_elo_df, 
            x='date', 
            y='Team_Elo',
            title=f"Historical Elo Rating for {selected_team_elo}",
            labels={'date': 'Date', 'Team_Elo': 'Elo Rating'},
            color_discrete_sequence=['#C8102E'] # NBA Red
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Helvetica Neue", color="#1D428A"),
            xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            yaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
