"""
Stock Market Trend Bot - Configuration

Sector ETFs + asset class ETFs vs SPY rolling Sortino ratio comparison.
"""

# Benchmark
BENCHMARK = "SPY"

# Sector ETFs
SECTOR_ETFS = {
    # ── Core S&P Sectors ──
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Utilities": "XLU",

    # ── Mag 7 & Big Tech ──
    "Mag 7": "MAGS",

    # ── Thematic / Industry ──
    "Semiconductors": "SMH",
    "Biotech": "XBI",
    "Genomics": "ARKG",
    "Cybersecurity": "CIBR",
    "AI & Robotics": "BOTZ",
    "Cloud Computing": "SKYY",
    "Quantum Computing": "QTUM",
    "Aerospace & Defense": "ITA",
    "Space": "UFO",
    "Clean Energy": "ICLN",
    "Solar": "TAN",
    "Homebuilders": "XHB",
    "Infrastructure": "PAVE",
    "Digital Infrastructure": "SRVR",
    "Water": "PHO",
    "Regional Banks": "KRE",
    "Fintech": "FINX",
    "E-Commerce": "IBUY",
    "Social Media": "SOCL",
    "Electric Vehicles": "DRIV",
    "3D Printing": "PRNT",
    "Gaming & Esports": "HERO",
    "Ark Innovation": "ARKK",
    "Internet": "FDN",
    "Software": "IGV",
    "Medtech": "IHI",
    "Insurance": "IAK",
    "Transports": "IYT",
    "Airlines": "JETS",
    "Food & Bev": "PBJ",
    "Agriculture": "MOO",
    "Retail": "XRT",
    "Leisure & Entertainment": "PEJ",
    "Rare Earth & Critical Minerals": "REMX",
    "MLPs & Pipelines": "AMLP",
    "IPO & New Listings": "IPO",
    "Mortgage REITs": "REM",
    "Shipping": "BOAT",
    "Timber & Forestry": "WOOD",

    # ── Commodities ──
    "Gold": "GLD",
    "Silver": "SLV",
    "Platinum": "PPLT",
    "Oil": "USO",
    "Natural Gas": "UNG",
    "Agriculture Commodities": "DBA",
    "Wheat": "WEAT",
    "Corn": "CORN",
    "Uranium": "URA",
    "Lithium & Battery": "LIT",
    "Copper Miners": "COPX",
    "Steel": "SLX",

    # ── Fixed Income ──
    "Bonds (20Y Treasury)": "TLT",
    "Bonds (Agg)": "AGG",
    "High Yield Bonds": "HYG",
    "TIPS (Inflation)": "TIP",
    # Treasury maturity curve (rotate duration)
    "T-Bills (0-3M)": "BIL",
    "T-Bills (Short)": "SHV",
    "Treasury 1-3Y": "SHY",
    "Treasury 3-7Y": "IEI",
    "Treasury 7-10Y": "IEF",
    # Credit / other fixed income
    "IG Corporates": "LQD",
    "Floating Rate": "FLOT",
    "Munis": "MUB",
    "Convertibles": "CWB",

    # ── Broad Market Indices ──
    "S&P 500": "IVV",          # IVV (not SPY, which is the benchmark) so it's a distinct rotatable sleeve
    "Dow Jones": "DIA",
    "Mid Cap": "IJH",
    "Total Market": "VTI",
    "Russell 1000 / Large Cap": "IWB",
    "Equal-Weight S&P": "RSP",

    # ── World Indices (single-country + regional; accel ranking self-cleans redundant sleeves) ──
    "Canada": "EWC",
    "France": "EWQ",
    "Italy": "EWI",
    "Spain": "EWP",
    "Switzerland": "EWL",
    "Netherlands": "EWN",
    "Sweden": "EWD",
    "Belgium": "EWK",
    "Austria": "EWO",
    "Ireland": "EIRL",
    "Norway": "NORW",
    "Denmark": "EDEN",
    "Hong Kong": "EWH",
    "Singapore": "EWS",
    "Israel": "EIS",
    "New Zealand": "ENZL",
    "Eurozone": "EZU",
    "Taiwan": "EWT",
    "Malaysia": "EWM",
    "Thailand": "THD",
    "Indonesia": "EIDO",
    "Philippines": "EPHE",
    "Vietnam": "VNM",
    "South Africa": "EZA",
    "Turkey": "TUR",
    "Saudi Arabia": "KSA",
    "Poland": "EPOL",
    "Chile": "ECH",
    "Peru": "EPU",
    "Colombia": "GXG",
    "Argentina": "ARGT",
    "Greece": "GREK",
    "Qatar": "QAT",
    "UAE": "UAE",
    "Pakistan": "PAK",
    "ACWI (All-Country World)": "ACWI",
    "World ex-US": "VEU",
    "Asia ex-Japan": "AAXJ",
    "Pacific ex-Japan": "EPP",
    "EM ex-China": "EMXC",
    "Frontier Markets": "FM",

    # ── Factor / Style ──
    "Dividend": "SCHD",
    "Momentum": "MTUM",
    "Value": "VTV",
    "Growth": "VUG",
    "Low Volatility": "SPLV",

    # ── Crypto (Bitcoin only) ──
    "Bitcoin ETF": "IBIT",
    "Bitcoin": "BTC-USD",

    # ── International ──
    "Emerging Markets": "EEM",
    "International (EAFE)": "EFA",
    "China": "MCHI",
    "India": "INDA",
    "Japan": "EWJ",
    "South Korea": "EWY",
    "Latin America": "ILF",
    "Brazil": "EWZ",
    "Mexico": "EWW",
    "Europe": "VGK",
    "Germany": "EWG",
    "UK": "EWU",
    "Australia": "EWA",
    "Africa": "AFK",

    # ── Size ──
    "Small Cap": "IWM",
    "Micro Cap": "IWC",
    "Nanotechnology": "TINY",
    "Nasdaq 100": "QQQ",

    # ── Alts ──
    "Cannabis": "MSOS",
}

# Data settings
DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"

# Rolling Sortino window (trading days)
SORTINO_WINDOW = 10

# Annualization factor (trading days per year)
TRADING_DAYS = 252

# Risk-free rate (annualized, approximate)
RISK_FREE_RATE = 0.05
