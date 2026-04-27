"""Classify markets using structural market characteristics rather than keywords."""

from app.data_client import Market

# Ordered: first match wins. Broad terms (e.g. "win") are placed after
# specific ones so they don't shadow more specific categories.
_TOPIC_RULES: list[tuple[str, list[str]]] = [
    # cricket MUST precede soccer — "Indian Premier League" contains "premier league"
    ("cricket",    ["cricket", "ipl", "indian premier league", "t20", "odi",
                    "test match", "super giants", "royal challengers",
                    "punjab kings", "psl", "islamabad", "lahore",
                    "hyderabad kingsmen"]),
    ("tennis",     ["tennis", "atp", "wta", "open:", "madrid open", "roland garros",
                    "wimbledon", "us open", "australian open", "davis cup",
                    "challenger", "itf ", "set handicap", "oeiras", "savannah",
                    "rome:", "barcelona open", "bmw open", "porsche tennis"]),
    ("soccer",     ["fc ", " fc", "football", "soccer", "premier league", "la liga",
                    "bundesliga", "serie a", "ligue 1", "champions league", "europa",
                    "mls", "eredivisie", "j-league", "k league", "a-league",
                    "world cup", "uefa", "copa", "draw?", "relegation",
                    "san lorenzo", "cienciano", "pumas", "whitecaps", "perth glory",
                    "melbourne city", "yokohama", "brentford", "fulham", "torino",
                    "aston villa", "sassuolo", "fiorentina"]),
    ("baseball",   ["mlb", "baseball", "world series", "yankees", "dodgers", "red sox",
                    "cubs", "astros", "mets", "giants vs", "mariners", "padres",
                    "rockies", "brewers", "white sox", "tigers vs", "athletics"]),
    ("basketball", ["nba", "basketball", "lakers", "celtics", "warriors", "bulls",
                    "heat vs", "nets vs", "bucks", "nuggets", "76ers", "cba ",
                    "jilin", "beijing ducks", "guangdong", "liaoning", "ningbo",
                    "fujian", "guangzhou", "jiangsu dragons", "shenzhen leopards",
                    "zhejiang lions", "qingdao eagles"]),
    ("esports",    ["esports", "dota", "csgo", "cs2", "valorant", "league of legends",
                    "lol ", "overwatch", "starcraft", "fortnite", "pubg", "rainbow six"]),
    ("politics",   ["election", "president", "congress", "senate", "parliament",
                    "minister", "referendum", "vote", "trump", "biden", "harris",
                    "democrat", "republican", "legislation", "ceasefire", "sanction",
                    "nato", "ukraine", "iran", "israel", "redistrict", "amendment"]),
    ("crypto",     ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol ",
                    "binance", "coinbase", "defi", "nft", "altcoin", "stablecoin"]),
    ("finance",    ["stock", "s&p", "nasdaq", "dow jones", "fed rate", "interest rate",
                    "gdp", "inflation", "recession", "ipo", "earnings", "oil price",
                    "gold price", "forex"]),
    ("entertainment", ["netflix", "oscar", "grammy", "emmy", "box office", "movie",
                       "album", "chart", "billboard", "spotify", "youtube",
                       "streaming", "series", "season"]),
]


def get_topic_category(question: str) -> str:
    """Return a broad topic category for a market question."""
    q = question.lower()
    for category, keywords in _TOPIC_RULES:
        if any(kw in q for kw in keywords):
            return category
    return "other"


def classify_market(market: Market) -> tuple[str, str]:
    """Classify a market by price, volume, and liquidity quality."""
    if market.yes_price < 0.02 or market.yes_price > 0.98:
        return ("excluded", "extreme probability")
    if market.volume < 50000:
        return ("excluded", "low volume")
    if market.liquidity_depth < 1000:
        return ("excluded", "low liquidity")

    if (
        market.volume >= 200000
        and market.liquidity_depth >= 10000
        and 0.10 <= market.yes_price <= 0.90
    ):
        return ("core", "high-quality market")

    return ("novelty", "medium-quality or uncertain market")


if __name__ == "__main__":
    sample = Market(
        market_id="test",
        question="Will BTC hit 100k?",
        yes_price=0.6,
        no_price=0.4,
        volume=150000,
        liquidity_depth=5000,
    )
    print(classify_market(sample))
