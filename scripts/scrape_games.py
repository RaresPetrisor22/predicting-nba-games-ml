from src.scraping.parser import make_games_csv
from src.scraping.scraper import get_games

def scrape_games():
    get_games()
    make_games_csv()

if __name__ == "__main__":
    scrape_games()