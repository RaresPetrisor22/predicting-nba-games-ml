
import os
from bs4 import BeautifulSoup
import time
import urllib.request
from datetime import datetime, timedelta

CURRENT_TIME = datetime.now()
SEASONS = [x for x in range(2016,CURRENT_TIME.year+1)]
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR,"standings")
SCORES_DIR = os.path.join(DATA_DIR,"scores")
ACTIVE_SEASON = CURRENT_TIME.year if CURRENT_TIME.month < 7 else CURRENT_TIME.year+1

def get_html(url,selector, sleep_time = 5, retries=3):
    html = None
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    for i in range(1, retries+1):
        time.sleep(sleep_time*i)
        try:
            req = urllib.request.Request(url,headers=headers)
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')

                soup = BeautifulSoup(html, "html.parser")
                element = soup.select_one(selector)
                if element:
                    return str(element)
                else:
                    print(f"Selector '{selector}' not found on {url}")
                    return None
        
        except Exception as e:
            print(f"Attempt {i} failed for {url}: {e}")
            if i < retries:
                print(f"Retrying in {sleep_time*i} seconds...")
            else:
                print("Max retries reached")
                return None

def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = get_html(url, "#content .filter")

    if not html:
        return
    soup = BeautifulSoup(html,features="lxml")
    links = soup.find_all("a")
    href = [l["href"] for l in links]
    standings_pages = [f"https://www.basketball-reference.com{l}" for l in href]

    now = datetime.now()
    current_month = now.strftime("%B").lower() 
    
    first_day_this_month = now.replace(day=1)
    prev_month = (first_day_this_month - timedelta(days=1)).strftime("%B").lower()
    
  
    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])

        month_in_url = url.split("-")[-1].replace(".html", "")
        
        force_rescrape = (season == ACTIVE_SEASON) and (month_in_url in [current_month, prev_month])

        if os.path.exists(save_path) and not force_rescrape:
            continue
        
        print(f"Scraping URL: {url}")

        html = get_html(url, "#div_schedule")
        
        if html:
            with open(save_path, "w+", encoding="utf-8") as f:
                f.write(html)
        

def scrape_game(standings_file):
    with open(standings_file,'r') as f:
        html = f.read()

    soup = BeautifulSoup(html,features="lxml")
    links = soup.find_all("a")
    hrefs = [l.get("href") for l in links]
    box_scores = [l for l in hrefs if l and "boxscores" in l and ".html" in l]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]
    for url in box_scores:
        save_path = os.path.join(SCORES_DIR,url.split("/")[-1])
        if os.path.exists(save_path):
            continue
        
        print(f"Scraping game: {url.split("/")[-1]}")

        html = get_html(url,"#content")
        if not html:
            continue

        with open(save_path,"w+",encoding="utf-8") as f:
            f.write(html)


def get_games():
    # if running scraper for the first time, all seasons should be scraped, not just the active one
    scrape_season(ACTIVE_SEASON)
    standings_files = os.listdir(STANDINGS_DIR)

    for f in standings_files:
        filepath = os.path.join(STANDINGS_DIR,f)
        scrape_game(filepath)



