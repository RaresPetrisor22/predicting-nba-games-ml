import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt

# Import custom feature engineering functions
from src.features.feature_engineer import load_raw_data
from scripts.predict_tonight import predict_tonight
from src.model.feature_importance import get_feature_importance

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
        font-size: 2.4rem;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8; /* Cool Gray */
        font-size: 1.3rem;
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
    /* Style the little "x" (close) button inside the pill too, to match */
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] svg {
        fill: #FCBF49 !important;
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
            <h1 style="margin: 0; padding: 0; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 700; color: #FFFFF0;">
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

    @st.cache_data()
    def get_cached_predictions():
        preds_df = pd.read_csv("data/predictions.csv")
        new_games = preds_df[preds_df['actual'] == -1]
        return new_games.to_dict(orient="records")
    
    df = get_cached_data("data/nba_games_processed.csv")
    all_teams = sorted(list(df['team'].unique()))

    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Tonight's Predictions", 
        "Team Analytics", 
        "Historical Elo Tracker",
        "Model Performance"
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
                home_abbr = match['team']
                away_abbr = match['team_opp']
                home_info = NBA_TEAMS.get(home_abbr, {'name': home_abbr, 'id': ''})
                away_info = NBA_TEAMS.get(away_abbr, {'name': away_abbr, 'id': ''})
                
                # Determine expected winner for inline display
                is_home_winner = float(match['home_prob_win']) > float(match['away_prob_win'])
                
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
            <div style="color: #FCBF49; font-size: 2.2rem; font-weight: bold;">{float(match['home_prob_win']):.2%}</div>
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
        <div style="color: #FCBF49; font-size: 2.2rem; font-weight: bold;">{float(match['away_prob_win']):.2%}</div>
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
        advanced_stats = ['date','team','team_opp',"won",'ts%','efg%','3par','ftr','orb%','drb%','trb%','ast%','stl%','blk%','tov%','usg%','ortg','drtg']

        stat_groups = {
            "Basic": basic_stats,
            "Advanced": advanced_stats
        }
        
        team_games = data[data['team'] == selected_team].copy()
        team_games = team_games.sort_values(by='date', ascending=False)
        
        last_10 = team_games.head(10)
        
        for stat_type, stats_list in stat_groups.items():
            st.markdown(
                f"<h3 style='margin-bottom: 8px;'>Last 10 Games "
                f"<span style='color:#FCBF49;'>{stat_type} Stats </span> "
                f"for <em>{NBA_TEAMS.get(selected_team, {}).get('name', selected_team)}</em></h3>",
                unsafe_allow_html=True
            )

            # Keep only columns that exist in this slice
            display_cols = [c for c in stats_list if c in last_10.columns]

            # --- UI copy: never mutate the source data ---
            ui_df = last_10[display_cols].copy()

            # Build column_config incrementally
            col_cfg = {}

            # Transform team logo columns → SVG URL strings for ImageColumn
            for logo_col, title in [("team", "Team"), ("team_opp", "Opp")]:
                if logo_col in ui_df.columns:
                    ui_df[logo_col] = ui_df[logo_col].apply(
                        lambda abbr: get_logo_url(NBA_TEAMS.get(str(abbr).strip(), {}).get("id", ""))
                    )
                    col_cfg[logo_col] = st.column_config.ImageColumn(title, width="small")

            # Transform won column → readable colored text for TextColumn
            if "won" in ui_df.columns:
                def _won_label(v):
                    try:
                        return "✅ Win" if int(float(v)) == 1 else "❌ Loss"
                    except (ValueError, TypeError):
                        return str(v)
                ui_df["won"] = ui_df["won"].apply(_won_label)
                col_cfg["won"] = st.column_config.TextColumn("Result", width="small")

            numeric_cols = [c for c in ui_df.columns if ui_df[c].dtype in ['float64', 'int64']
                            and c not in ('won',)]

            styled_df = ui_df.style.format(
                {c: "{:.2f}" for c in numeric_cols},
                na_rep="—"
            )

            st.dataframe(
                styled_df,
                column_config=col_cfg,
                width='stretch',
                hide_index=True,
            )

        

    # --- TAB 3: Historical Elo Tracker ---
    with tab3:
        st.header("Historical Elo Tracker")
        
        selected_teams_elo = st.multiselect("Select Teams to Compare:", all_teams, 
                                            default=['BOS', 'LAL'], # Default to a classic rivalry
                                            format_func=lambda x: NBA_TEAMS[x]["name"], 
                                            key="team_select_elo")
        
        if not selected_teams_elo:
            st.warning("Please select at least one team to view the Elo chart.")
        else:
            team_dfs = []
            
            for team in selected_teams_elo:
                team_filter = (df['team'] == team) | (df['team_opp'] == team)
                team_elo_df = df[team_filter].copy()
                
                # A helper to pull the correct rating from the perspective of the chosen team
                def get_team_elo(row, t=team):
                    if row['team'] == t:
                        return row['home_elo']
                    else:
                        return row['away_elo']
                        
                team_elo_df['Team_Elo'] = team_elo_df.apply(get_team_elo, axis=1)
                team_elo_df['Team_Name'] = NBA_TEAMS.get(team, {}).get('name', team)
                team_elo_df = team_elo_df.sort_values('date')
                team_dfs.append(team_elo_df[['date', 'Team_Elo', 'Team_Name']])
                
            combined_elo_df = pd.concat(team_dfs)

            # Map of symbolic NBA Primary Colors
            TEAM_COLORS = {
                "Atlanta Hawks": "#E03A3E", "Boston Celtics": "#007A33", "Brooklyn Nets": "#FFFFFF",
                "Charlotte Hornets": "#1D1160", "Chicago Bulls": "#CE1141", "Cleveland Cavaliers": "#860038",
                "Dallas Mavericks": "#00538C", "Denver Nuggets": "#0E2240", "Detroit Pistons": "#C8102E",
                "Golden State Warriors": "#1D428A", "Houston Rockets": "#CE1141", "Indiana Pacers": "#FDBB30",
                "Los Angeles Clippers": "#C8102E", "Los Angeles Lakers": "#552583", "Memphis Grizzlies": "#5D76A9",
                "Miami Heat": "#98002E", "Milwaukee Bucks": "#00471B", "Minnesota Timberwolves": "#0C2340",
                "New Orleans Pelicans": "#0C2340", "New York Knicks": "#F58426", "Oklahoma City Thunder": "#007AC1",
                "Orlando Magic": "#0077C0", "Philadelphia 76ers": "#006BB6", "Phoenix Suns": "#1D1160",
                "Portland Trail Blazers": "#E03A3E", "Sacramento Kings": "#5A2D81", "San Antonio Spurs": "#C4CED4",
                "Toronto Raptors": "#CE1141", "Utah Jazz": "#002B5C", "Washington Wizards": "#002B5C"
            }

            # Plotting
            fig = px.line(
                combined_elo_df, 
                x='date', 
                y='Team_Elo',
                color='Team_Name',
                title="Historical Elo Rating Comparison",
                labels={'date': 'Date', 'Team_Elo': 'Elo Rating', 'Team_Name': 'Team'},
                color_discrete_map=TEAM_COLORS
            )
            
            fig.update_layout(
                plot_bgcolor='#1E293B',
                paper_bgcolor='#1E293B',
                font=dict(family="Helvetica Neue", color="#F8FAFC"),
                xaxis=dict(showgrid=True, gridcolor='#334155'),
                yaxis=dict(showgrid=True, gridcolor='#334155'),
                hovermode="closest",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title=""
                )
            )
            
            st.plotly_chart(fig, width='stretch')

        # --- Elo Methodology Explanation ---
        st.markdown(
"""
<div style="margin-top: 32px; background-color: #1E293B; border: 1px solid #334155;
            border-radius: 12px; padding: 28px 32px;">

<h3 style="color: #F8FAFC; margin: 0 0 6px 0; font-size: 1.2rem; letter-spacing: 0.5px;">
    How is the Elo Rating calculated?
</h3>
<p style="color: #94A3B8; font-size: 0.9rem; margin: 0 0 24px 0;">
    This model uses a custom Elo system adapted from chess, tuned for NBA basketball.
    Each team starts every season at <span style="color:#FCBF49; font-weight:bold;">1500</span> (with a partial regression toward that baseline between seasons).
</p>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">

<div style="background:#0F172A; border-radius:8px; padding:18px; border-left: 3px solid #FCBF49;">
<div style="color:#FCBF49; font-size:0.75rem; font-weight:bold; letter-spacing:1px; margin-bottom:8px;">STEP 1 — EXPECTED SCORE</div>
<code style="color:#F8FAFC; font-size:0.85rem;">E_home = 1 / (1 + 10<sup>((away_elo − home_elo) / 400)</sup>)</code>
<p style="color:#94A3B8; font-size:0.82rem; margin:10px 0 0 0;">
    The classic logistic curve gives each team a win probability before the game is played.
    A 400-point Elo gap = ~91% win probability for the stronger team.
</p>
</div>

<div style="background:#0F172A; border-radius:8px; padding:18px; border-left: 3px solid #1D428A;">
<div style="color:#4B9FE1; font-size:0.75rem; font-weight:bold; letter-spacing:1px; margin-bottom:8px;">STEP 2 — MARGIN-OF-VICTORY K-FACTOR</div>
<code style="color:#F8FAFC; font-size:0.85rem; line-height:1.8;">K = 20 × (|MOV| + 3)<sup>0.8</sup> / (7.5 + 0.006 × |Δelo|)</code>
<p style="color:#94A3B8; font-size:0.82rem; margin:10px 0 0 0;">
    Unlike chess, NBA outcomes scale with the margin of victory (<strong style="color:#F8FAFC">MOV</strong>).
    Blowout wins earn more Elo than squeakers. The denominator dampens gains from beating already-weaker opponents by large margins.
</p>
</div>

<div style="background:#0F172A; border-radius:8px; padding:18px; border-left: 3px solid #1D428A;">
<div style="color:#4B9FE1; font-size:0.75rem; font-weight:bold; letter-spacing:1px; margin-bottom:8px;">STEP 3 — ELO UPDATE RULE</div>
<code style="color:#F8FAFC; font-size:0.85rem; line-height:1.8;">new_elo = old_elo + K × (actual − expected)</code>
<p style="color:#94A3B8; font-size:0.82rem; margin:10px 0 0 0;">
    <strong style="color:#F8FAFC">actual</strong> = 1 for a win, 0 for a loss.
    Beating a highly-rated opponent yields a large gain; losing to a weak team yields a large drop.
    Both teams' ratings are updated after every game.
</p>
</div>

<div style="background:#0F172A; border-radius:8px; padding:18px; border-left: 3px solid #FCBF49;">
<div style="color:#FCBF49; font-size:0.75rem; font-weight:bold; letter-spacing:1px; margin-bottom:8px;">STEP 4 — SEASON RESET</div>
<code style="color:#F8FAFC; font-size:0.85rem; line-height:1.8;">elo<sub>new season</sub> = elo × 0.75 + 1505 × 0.25</code>
<p style="color:#94A3B8; font-size:0.82rem; margin:10px 0 0 0;">
    At the start of each new season, all ratings regress 25% toward the mean (<strong style="color:#F8FAFC">1505</strong>).
    This accounts for roster changes and prevents ancient dominance from over-influencing current predictions.
</p>
</div>

</div>
</div>
""",
            unsafe_allow_html=True
        )

    # --- TAB 4: Model Performance ---
    with tab4:
        st.header("Model Performance")

        from src.model.model_metrics import get_model_metrics

        @st.cache_data(ttl=21600)
        def get_cached_model_metrics():
            return get_model_metrics()

        acc, loss, cm, prob_true, prob_pred = get_cached_model_metrics()

        kpi_col1, kpi_col2, kpi_col3 = st.columns([1, 1, 1.8])
        with kpi_col1:
            st.markdown(
                f"""
                <div style="background-color: #1E293B; border-radius: 10px; padding: 20px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3); height: 135px; display: flex; flex-direction: column; justify-content: center;">
                    <div style="color: #94A3B8; font-size: 1.1rem; margin-bottom: 8px;">Accuracy on the 2026 Regular Season</div>
                    <div style="color: #FCBF49; font-size: 2rem; font-weight: bold;">{acc * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True
            )
        with kpi_col2:
            st.markdown(
                f"""
                <div style="background-color: #1E293B; border-radius: 10px; padding: 20px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3); height: 135px; display: flex; flex-direction: column; justify-content: center;">
                    <div style="color: #94A3B8; font-size: 1.1rem; margin-bottom: 8px;">Log Loss (Cross-entropy)</div>
                    <div style="color: #FCBF49; font-size: 2rem; font-weight: bold;">{loss:.3f}</div>
                </div>
                """, unsafe_allow_html=True
            )
        with kpi_col3:
            num_training_games = len(df[df['season'] < 2026])
            num_features = len(df.columns)
            st.markdown(
                f"""
                <div style="background-color: #1E293B; border-radius: 10px; padding: 18px 20px; border: 1px solid #334155; border-left: 4px solid #FCBF49; box-shadow: 0 4px 6px rgba(0,0,0,0.3); height: 135px; display: flex; flex-direction: column; justify-content: center;">
                    <div style="color: #F8FAFC; font-size: 0.98rem; line-height: 1.55;">
                        Trained on data from the <strong style="color: #FCBF49;">2015 to 2025</strong> seasons 
                        (totaling <strong style="color: #FCBF49;">{num_training_games:,}</strong> games from the home team perspective, using <strong style="color: #FCBF49;">{num_features}</strong> features), 
                        making out-of-sample predictions on the <strong style="color: #FCBF49;">2026</strong> season.
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_cm = px.imshow(
                cm, 
                text_auto=True, 
                aspect="auto", 
                labels=dict(x="Model prediction", y="Actual outcome", color="Count"),
                x=['Loss (0)', 'Win (1)'],
                y=['Loss (0)', 'Win (1)'],
                title="Confusion Matrix",
                color_continuous_scale=['#F8FAFC', '#FEF08A', '#D4AF37']
            )
            fig_cm.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F8FAFC'),
                title_font=dict(color='#F8FAFC', size=16),
                coloraxis_colorbar=dict(title="Count", tickfont=dict(color="#F8FAFC")),
                xaxis=dict(showgrid=False, linecolor='#94A3B8'),
                yaxis=dict(showgrid=False, linecolor='#94A3B8')
            )
            st.plotly_chart(fig_cm, width='stretch')

        with chart_col2:
            import plotly.graph_objects as go
            fig_cal = go.Figure()
            
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], 
                mode='lines', 
                line=dict(dash='dash', color='#94A3B8'), 
                name='Perfectly Calibrated'
            ))
            
            fig_cal.add_trace(go.Scatter(
                x=prob_pred, y=prob_true, 
                mode='lines+markers', 
                line=dict(color='#D4AF37', width=3), 
                marker=dict(size=8, color='#D4AF37'), 
                name='NBA Model'
            ))
            
            fig_cal.update_layout(
                title=dict(text="Calibration Curve", font=dict(color='#F8FAFC', size=16)),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F8FAFC'),
                xaxis=dict(title="Average Predicted Probability", showgrid=True, gridcolor='#334155', linecolor='#94A3B8', zeroline=False),
                yaxis=dict(title="True Fraction of Wins", showgrid=True, gridcolor='#334155', linecolor='#94A3B8', zeroline=False),
                legend=dict(font=dict(color='#F8FAFC'), bgcolor='rgba(0,0,0,0)', yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_cal, width='stretch')

        st.markdown("<hr style='border: 1px solid #334155; margin: 30px 0;'>", unsafe_allow_html=True)

        @st.cache_data(ttl=21600)
        def get_cached_feature_importance():
            return get_feature_importance()
        
        feature_importance_df = get_cached_feature_importance()
        
        if feature_importance_df is not None and not feature_importance_df.empty:
            feature_importance_df = feature_importance_df.sort_values(by='Abs_Weight', ascending=True)
            

            fig = px.bar(
                feature_importance_df, 
                x='Weight', 
                y='Feature', 
                orientation='h',
                color='Weight',
                color_continuous_scale=[
                    [0.00, '#ef4444'],  
                    [0.25, '#b91c1c'],  
                    [0.49, '#7f1d1d'],  
                    [0.51, '#14532d'],  
                    [0.75, '#15803d'],  
                    [1.00, '#22c55e']   
                ],
                color_continuous_midpoint=0
            )
            fig.update_coloraxes(showscale=False)
            
            fig.update_traces(marker_line_width=0)
            
            fig.update_layout(
                plot_bgcolor='#0F172A',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F8FAFC'),
                showlegend=False,
                title="Top 15 Feature Importances",
                xaxis_title="Weight",
                yaxis_title="Feature",
                xaxis=dict(showgrid=True, gridcolor='#334155', gridwidth=1),
                yaxis=dict(showgrid=False)
            )
            
            fig.add_vline(x=0, line_dash="dash", line_color="#F8FAFC", line_width=1.5, opacity=0.7)
            
            st.plotly_chart(fig, width='stretch')

            with st.expander("Feature Naming Legend"):
                st.write("**`_opp_roll10`**: Stats the team allowed against their opponents. (e.g. last 10 games average for opponent points allowed)")
                st.write("**`_roll10_opp_history`**: Averages for the opposing team they are facing.")


if __name__ == "__main__":
    main()
