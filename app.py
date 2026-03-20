import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Import custom feature engineering functions
from src.features.feature_engineer import load_raw_data
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
    page_icon="assets/basketball-logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR NBA THEME ---
st.markdown("""
    <style>
    /* Main body background */
    .stApp {
        background-color: #0F172A;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E293B;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94A3B8;
        border: 1px solid #334155;
        border-bottom: none;
        width: 160px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1D428A;
        color: #F8FAFC !important;
        border-top: 4px solid #1D428A; /* Muted blue border */
    }
    .stTabs [data-baseweb="tab-highlight"] {
        height: 3.5px;
        background-color: #1D428A;
        margin-left: 1px;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #FCBF49; 
        font-size: 2rem;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8; /* Cool Gray */
        font-size: 1.1rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        border: 1px solid #334155;
        border-top: 4px solid #1D428A;
        margin-bottom: 15px;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #F8FAFC;
        font-weight: bold;
    }
    /* Hide the entire top toolbar (hamburger menu, deploy button, etc.) */
    [data-testid="stToolbar"] {
        visibility: hidden !important;
        display: none !important;
    }

    /* Hide the colored decoration bar at the very top of the screen */
    [data-testid="stDecoration"] {
        visibility: hidden !important;
        display: none !important;
    }

    </style>
""", unsafe_allow_html=True)


# --- MAIN APP ---
def main():
    # Load and encode logo for sharp, centered display
    import base64
    def get_base64_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    logo_b64 = get_base64_image("assets/basketball-logo.png")
    
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 18px; margin-bottom: 25px;">
            <img src="data:image/png;base64,{logo_b64}" style="width: 48px; height: 48px; object-fit: contain;">
            <h1 style="margin: 0; padding: 0; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 700; color: #F8FAFC;">
                Basketball ML Prediction Dashboard
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    **An automated, end-to-end Machine Learning pipeline for forecasting NBA matchups.** This dashboard serves **live win probabilities** powered by a Logistic Regression model. Behind the scenes, a cloud-hosted web scraper updates the database daily, engineering custom features like **10-game rolling averages** and **dynamic Elo ratings** to predict tonight's games.
    """)

    @st.cache_data
    def get_cached_data(path):
        df = load_raw_data(path)
        return df

    @st.cache_data(ttl=21600)
    def get_cached_predictions():
        return predict_tonight()
    
    df = get_cached_data("data/nba_games_processed.csv")
    all_teams = sorted(list(df['team'].unique()))

    # Create Tabs
    tab1, tab2, tab3 = st.tabs([
        "Tonight's Predictions", 
        "Team Analytics", 
        "Historical Elo Tracker"
    ])

    # --- TAB 1: Tonight's Predictions ---
    with tab1:
        st.header("Tonight's Matchups")
        st.markdown(f"**Date:** {datetime.now().strftime('%B %d, %Y')}")
        
        with st.spinner("Fetching tonight's matchups and running predictions..."):
            matchups = get_cached_predictions()

        # Key stats to surface in the breakdown (roll10 column suffix → display name)
        KEY_STATS = [
            ('pts_roll10',   'Points'),
            ('fg%_roll10',   'FG%'),
            ('3p%_roll10',   '3P%'),
            ('ft%_roll10',   'FT%'),
            ('ast_roll10',   'Assists'),
            ('trb_roll10',   'Rebounds'),
            ('stl_roll10',   'Steals'),
            ('blk_roll10',   'Blocks'),
            ('tov_roll10',   'Turnovers'),
            ('ortg_roll10',  'Off. Rating'),
            ('drtg_roll10',  'Def. Rating'),
            ('won_roll10',   'Win Rate Percentage'),
        ]

        def get_team_last10_stats(team_abbr):
            """Return latest row of roll10 stats for a team."""
            team_rows = df[df['team'] == team_abbr].sort_values('date')
            if team_rows.empty:
                return {}
            latest = team_rows.iloc[-1]
            return {col: latest[col] for col, _ in KEY_STATS if col in latest.index}

        if not matchups:
            st.info("No games scheduled for today. See you tomorrow!")
        else:
            for match in matchups:
                home_abbr = match['home']
                away_abbr = match['away']
                home_info = NBA_TEAMS.get(home_abbr, {'name': home_abbr, 'id': ''})
                away_info = NBA_TEAMS.get(away_abbr, {'name': away_abbr, 'id': ''})
                
                # Determine expected winner for inline display
                is_home_winner = float(match['home_prob']) > float(match['away_prob'])
                
                # Side-aware styles for the winner badge
                winner_bg_rgba = "rgba(214, 40, 40, 0.80)"
                badge_base_style = f"background-color: {winner_bg_rgba}; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: bold; vertical-align: middle; white-space: nowrap;"
                
                home_winner_badge = f'<span style="{badge_base_style} margin-left: 12px;">EXPECTED WINNER</span>'
                away_winner_badge = f'<span style="{badge_base_style} margin-right: 12px;">EXPECTED WINNER</span>'
                
                # Use dedented HTML to avoid Markdown treating it as a code block (due to leading spaces)
                match_html = f"""
<div class="metric-container" style="display: flex; justify-content: space-between; align-items: center; padding: 20px;">
<!-- Home Team -->
    <div style="display: flex; align-items: center; width: 40%;">
        <img src="{get_logo_url(home_info['id'])}" style="height: 70px; margin-right: 15px;">
        <div>
            <div style="color: #F8FAFC; font-size: 1.2rem; font-weight: bold; display: flex; align-items: center;">
                {home_info['name']} {home_winner_badge if is_home_winner else ''}
            </div>
            <div style="color: #FCBF49; font-size: 2.2rem; font-weight: bold;">{match['home_prob']}%</div>
        </div>
    </div>
    
<!-- VS -->
<div style="width: 20%; text-align: center;">
        <h3 style="color: #94A3B8; margin: 0; font-size: 2.5rem; letter-spacing: 2px;">VS</h3>
</div>
    
<!-- Away Team -->
<div style="display: flex; align-items: center; justify-content: flex-end; width: 40%; text-align: right;">
    <div>
        <div style="color: #F8FAFC; font-size: 1.2rem; font-weight: bold; display: flex; align-items: center; justify-content: flex-end;">
            {'' if is_home_winner else away_winner_badge} {away_info['name']}
        </div>
        <div style="color: #FCBF49; font-size: 2.2rem; font-weight: bold;">{match['away_prob']}%</div>
    </div>
    <img src="{get_logo_url(away_info['id'])}" style="height: 70px; margin-left: 15px;">
</div>
</div>
"""
                st.markdown(match_html, unsafe_allow_html=True)

                # Stats where LOWER is better
                LOWER_IS_BETTER = {'tov_roll10', 'drtg_roll10'}

                def stat_colors(col, home_val, away_val):
                    """Return (home_color, away_color) based on which value is better."""
                    GREEN = "#4ADE80"
                    GRAY  = "#94A3B8"
                    if home_val is None or away_val is None:
                        return GRAY, GRAY
                    if abs(float(home_val) - float(away_val)) < 0.01:
                        return GRAY, GRAY
                    lower_wins = col in LOWER_IS_BETTER
                    home_better = (home_val < away_val) if lower_wins else (home_val > away_val)
                    if home_better:
                        return GREEN, GRAY
                    else:
                        return GRAY, GREEN

                # --- Expandable stats dropdown ---
                with st.expander("Last 10 Games Averages"):
                    home_stats = get_team_last10_stats(home_abbr)
                    away_stats = get_team_last10_stats(away_abbr)

                    # Centered title inside the expander
                    st.markdown(
                        f"<div style='display:flex; justify-content:center; align-items:center; "
                        f"color:#F8FAFC; font-size:1.05rem; font-weight:bold; padding: 4px 0 16px 0; gap:16px;'>"
                        f"<img src='{get_logo_url(home_info['id'])}' style='height:32px; vertical-align:middle;'>"
                        f"<span>{home_info['name']}</span>"
                        f"<span></span>"
                        f"<span>{away_info['name']}</span>"
                        f"<img src='{get_logo_url(away_info['id'])}' style='height:32px; vertical-align:middle;'>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    col_home, col_mid, col_away = st.columns([5, 1, 5])

                    with col_home:
                        for col, label in KEY_STATS:
                            h_val = home_stats.get(col)
                            a_val = away_stats.get(col)
                            home_color, _ = stat_colors(col, h_val, a_val)
                            formatted = f"{h_val:.1f}" if h_val is not None else "N/A"
                            st.markdown(
                                f"<div style='display:flex; justify-content:space-between; padding:6px 10px; "
                                f"border-bottom:1px solid #334155;'>"
                                f"<span style='color:#94A3B8; font-size:0.9rem;'>{label}</span>"
                                f"<span style='color:{home_color}; font-weight:bold; font-size:0.9rem;'>{formatted}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    with col_mid:
                        # Centered vertical divider
                        st.markdown(
                            "<div style='display:flex; flex-direction:column; align-items:center; "
                            "height:100%; padding-top:40px;'>"
                            "<div style='width:2px; background:linear-gradient(to bottom, transparent, #334155, transparent); "
                            "flex:1; min-height:300px;'></div>"
                            "</div>",
                            unsafe_allow_html=True
                        )

                    with col_away:
                        for col, label in KEY_STATS:
                            h_val = home_stats.get(col)
                            a_val = away_stats.get(col)
                            _, away_color = stat_colors(col, h_val, a_val)
                            formatted = f"{a_val:.1f}" if a_val is not None else "N/A"
                            st.markdown(
                                f"<div style='display:flex; justify-content:space-between; padding:6px 10px; "
                                f"border-bottom:1px solid #334155;'>"
                                f"<span style='color:{away_color}; font-weight:bold; font-size:0.9rem;'>{formatted}</span>"
                                f"<span style='color:#94A3B8; font-size:0.9rem;'>{label}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )


    # --- TAB 2: Team Analytics ---
    with tab2:
        data = get_cached_data("data/nba_games.csv")
        st.header("Team Analytics")
        
        selected_team = st.selectbox("Select a Team:", all_teams, format_func=lambda x: NBA_TEAMS[x]["name"],key="team_select_analytics")
        
        basic_stats = ['date','team','team_opp', 'won','pts', 'pts_opp',"fg","fga","fg%","3p","3pa","3p%","ft","fta","ft%","orb","drb","trb","ast","stl","blk","tov","pf"] 
        advanced_stats = ['date','team','team_opp','ts%','efg%','3par','ftr','orb%','drb%','trb%','ast%','stl%','blk%','tov%','usg%','ortg','drtg']

        stat_groups = {
            "Basic": basic_stats,
            "Advanced": advanced_stats
        }
        
        team_games = data[data['team'] == selected_team].copy()
        team_games = team_games.sort_values(by='date', ascending=False)
        
        last_10 = team_games.head(10)
        
        for stat_type, stats_list in stat_groups.items():
            st.subheader(f"Last 10 Games {stat_type} Stats for *{selected_team}*")
            
            # Keep only columns that exist
            display_cols = [c for c in stats_list if c in last_10.columns]
            
            st.dataframe(
                last_10[display_cols].style.format(precision=2),
                width="stretch",
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
            color_discrete_sequence=['#FCBF49'] 
        )
        
        fig.update_layout(
            plot_bgcolor='#1E293B',
            paper_bgcolor='#1E293B',
            font=dict(family="Helvetica Neue", color="#F8FAFC"),
            xaxis=dict(showgrid=True, gridcolor='#334155'),
            yaxis=dict(showgrid=True, gridcolor='#334155'),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, width='stretch')

if __name__ == "__main__":
    main()
