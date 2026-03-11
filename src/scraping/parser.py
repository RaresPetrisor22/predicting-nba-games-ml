
import datetime
import os
import pandas as pd
from bs4 import BeautifulSoup, Comment
import io

SCORE_DIR = "data/scores"
CSV_PATH = "data/nba_games.csv"  

def parse_html(box_scores):
    with open(box_scores,encoding="utf-8") as f:
        html = f.read()
    
    soup = BeautifulSoup(html,features="lxml")
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]
    return soup


def read_line_score(soup):
    placeholder = soup.find(id="all_line_score")
    if placeholder:
        for comment in placeholder.find_all(string=lambda text: isinstance(text, Comment)):
            if 'id="line_score"' in comment:
                table_soup = BeautifulSoup(comment, "html.parser")
                html_string = str(table_soup)
                break
    else:
        html_string = str(soup)
        
    line_score = pd.read_html(io.StringIO(html_string), attrs = {"id": "line_score"})[0]
    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    line_score = line_score[["team","total"]]
    return line_score


def read_stats(soup,team,stat):
    df = pd.read_html(io.StringIO(str(soup)),attrs={"id": f"box-{team}-game-{stat}"},index_col=0)[0]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def get_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all("a")]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return int(season)


def find_new_games():
    parsed_ids = set()
    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH, usecols=["id"])
        parsed_ids = set(existing_df["id"].values)
        print(f"Found {len(parsed_ids)} games already in the CSV.")

    new_box_scores = []
    box_scores = os.listdir(SCORE_DIR)
    box_scores = [os.path.join(SCORE_DIR,f) for f in box_scores]
    
    for file_path in box_scores:
        game_id = os.path.basename(file_path)[:-5]
        if game_id not in parsed_ids:
            new_box_scores.append(file_path)

    print(f"Found {len(new_box_scores)} NEW games to parse.")

    if len(new_box_scores) == 0:
        print("Dataset is completely up to date.")
    return new_box_scores

def get_stats():
    base_cols = None
    games = []
    new_box_scores = find_new_games()
    for box_score in new_box_scores:
        soup = parse_html(box_score)
        line_score = read_line_score(soup)
        teams = list(line_score["team"])

        summaries = []
        for i,team in enumerate(teams):
            basic = read_stats(soup,team,"basic")
            advanced = read_stats(soup,team,"advanced")

            totals = pd.concat([basic.iloc[-1,:], advanced.iloc[-1,:]])
            totals.index = totals.index.str.lower()
            totals = totals.rename(i)
            if base_cols is None:
                base_cols = list(totals.index)
                base_cols = [b for b in base_cols if "bpm" not in b]
            
            summary = totals[base_cols]
            summaries.append(summary)

        summary = pd.concat(summaries, axis=1).T

        game = pd.concat([summary,line_score],axis=1)
        game = game.loc[:, ~game.columns.duplicated()]

        game["home"] = [0,1]
        game_opp = game.iloc[::-1].reset_index()
        game_opp.columns += "_opp"

        full_game = pd.concat([game,game_opp],axis=1)

        full_game["season"] = get_season_info(soup)

        full_game["date"] = os.path.basename(box_score)[:8]
        full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

        full_game["won"] = full_game["total"] > full_game["total_opp"]

        full_game["id"] = os.path.basename(box_score)[:-5]

        games.append(full_game)

        if len(games) % 100 == 0:
            print(f"{len(games)} / {len(box_scores)} ")
        
    return games


def make_games_csv():
    games = get_stats()
    if len(games) > 0:
        new_games_df = pd.concat(games, ignore_index=True)
        
        if os.path.exists(CSV_PATH):
            old_df = pd.read_csv(CSV_PATH)
            
            combined_df = pd.concat([old_df, new_games_df], ignore_index=True)
            
            combined_df.to_csv(CSV_PATH, index=False)
            print(f"Successfully aligned and added {len(new_games_df)} new games!")
        else:
            new_games_df.to_csv(CSV_PATH, index=False)
            print("Created new CSV file with parsed games.")






