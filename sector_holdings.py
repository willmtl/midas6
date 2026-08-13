"""
Stock Market Trend Bot - Sector ETF Holdings

Top holdings for each sector ETF, stored as arrays for downstream use.
Commodity ETFs (GLD, SLV, USO, UNG, TLT) hold physical assets / futures, not stocks.

Last updated: 2026-06-30
"""

HOLDINGS = {
    # ── Core S&P Sectors ──

    "Technology": {
        "etf": "XLK",
        "holdings": [
            "NVDA", "AAPL", "MSFT", "MU", "AVGO", "AMD", "INTC", "CSCO",
            "LRCX", "ORCL", "CRM", "ADBE", "NOW", "QCOM", "TXN", "SNPS",
            "CDNS", "KLAC", "AMAT", "PLTR",
        ],
    },
    "Healthcare": {
        "etf": "XLV",
        "holdings": [
            "LLY", "JNJ", "ABBV", "UNH", "MRK", "TMO", "AMGN", "GILD",
            "ISRG", "PFE", "ABT", "SYK", "MDT", "VRTX", "BSX", "REGN",
            "EW", "BDX", "DXCM", "ZTS",
        ],
    },
    "Energy": {
        "etf": "XLE",
        "holdings": [
            "XOM", "CVX", "COP", "SLB", "WMB", "VLO", "MPC", "EOG",
            "PSX", "BKR", "OXY", "HAL", "DVN", "HES", "KMI", "CTRA",
            "FANG", "OKE", "TRGP", "LNG",
        ],
    },
    "Financials": {
        "etf": "XLF",
        "holdings": [
            "BRK-B", "JPM", "V", "MA", "BAC", "GS", "MS", "WFC",
            "C", "AXP", "SCHW", "PGR", "BLK", "CB", "MMC", "ICE",
            "USB", "CME", "PNC", "TFC",
        ],
    },
    "Consumer Discretionary": {
        "etf": "XLY",
        "holdings": [
            "AMZN", "TSLA", "HD", "TJX", "MCD", "BKNG", "LOW", "SBUX",
            "MAR", "GM", "NKE", "ORLY", "AZO", "CMG", "ROST", "DHI",
            "LEN", "F", "YUM", "LULU",
        ],
    },
    "Consumer Staples": {
        "etf": "XLP",
        "holdings": [
            "WMT", "COST", "PG", "KO", "PM", "MDLZ", "MO", "CL",
            "PEP", "MNST", "GIS", "KHC", "HSY", "KDP", "SYY", "K",
            "STZ", "KR", "CLX", "TSN",
        ],
    },
    "Industrials": {
        "etf": "XLI",
        "holdings": [
            "CAT", "GE", "GEV", "RTX", "BA", "UNP", "ETN", "HON",
            "UBER", "DE", "TT", "LMT", "WM", "UPS", "FDX", "ITW",
            "NOC", "CSX", "GD", "NSC",
        ],
    },
    "Materials": {
        "etf": "XLB",
        "holdings": [
            "LIN", "NEM", "NUE", "FCX", "VMC", "CRH", "APD", "STLD",
            "CTVA", "SHW", "ECL", "PPG", "DD", "DOW", "MLM", "IFF",
            "PKG", "IP", "CE", "FMC",
        ],
    },
    "Real Estate": {
        "etf": "XLRE",
        "holdings": [
            "WELL", "PLD", "EQIX", "AMT", "SPG", "DLR", "PSA", "VTR",
            "CCI", "O", "EXR", "AVB", "IRM", "VICI", "SBAC", "ARE",
            "MAA", "UDR", "ESS", "KIM",
        ],
    },
    "Communication Services": {
        "etf": "XLC",
        "holdings": [
            "META", "GOOGL", "GOOG", "TTWO", "LYV", "ECHO", "DIS",
            "WBD", "EA", "OMC", "NFLX", "TMUS", "T", "VZ", "CHTR",
            "IPG", "NWSA", "PARA", "MTCH", "FOXA",
        ],
    },
    "Utilities": {
        "etf": "XLU",
        "holdings": [
            "NEE", "SO", "DUK", "CEG", "AEP", "SRE", "D", "VST",
            "ETR", "XEL", "EXC", "WEC", "ES", "AWK", "PPL", "ED",
            "EIX", "AES", "DTE", "FE",
        ],
    },

    # ── Mag 7 ──

    "Mag 7": {
        "etf": "MAGS",
        "holdings": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        ],
    },

    # ── Thematic / Industry ──

    "Semiconductors": {
        "etf": "SMH",
        "holdings": [
            "NVDA", "TSM", "MU", "AMD", "INTC", "AVGO", "QCOM", "TXN",
            "LRCX", "KLAC", "AMAT", "MRVL", "ADI", "SNPS", "CDNS",
            "ON", "MCHP", "NXPI", "ASML", "SWKS",
        ],
    },
    "Biotech": {
        "etf": "XBI",
        "holdings": [
            "TVTX", "RVMD", "ALKS", "TWST", "TGTX", "ARWR", "BEAM",
            "NBIX", "KRYS", "EXEL", "MRNA", "BMRN", "SGEN", "IONS",
            "PCVX", "HALO", "PTCT", "CORT", "INSM", "RARE",
        ],
    },
    "Genomics": {
        "etf": "ARKG",
        "holdings": [
            "TWST", "CRSP", "TEM", "PSNL", "TXG", "ABSI", "BEAM",
            "GH", "ILMN", "NTRA", "PACB", "NVTA", "CDNA", "EXAI",
            "RXRX", "VEEV", "NTLA", "BNR", "OLINK", "FATE",
        ],
    },
    "Cybersecurity": {
        "etf": "CIBR",
        "holdings": [
            "CRWD", "PANW", "FTNT", "CSCO", "AVGO", "NET", "AKAM",
            "DDOG", "OKTA", "FFIV", "ZS", "S", "TENB", "RPD", "VRNS",
            "QLYS", "CYBR", "SAIL", "MNDT", "CHKP",
        ],
    },
    "AI & Robotics": {
        "etf": "BOTZ",
        "holdings": [
            "6861.T", "ABBN.SW", "6954.T", "NVDA", "ISRG", "300124.SZ",
            "6273.T", "6383.T", "300757.SZ", "6506.T", "ROK", "BRKS",
            "TER", "IRBT", "KUKA.DE", "ZBRA", "CGNX", "UPST", "PATH",
            "NNDM",
        ],
    },
    "Cloud Computing": {
        "etf": "SKYY",
        "holdings": [
            "DOCN", "ORCL", "DELL", "NTNX", "AMZN", "LUMN", "IBM",
            "P", "GOOGL", "ANET", "MSFT", "CRM", "SNOW", "MDB",
            "DDOG", "NET", "ZS", "ESTC", "CFLT", "HUBS",
        ],
    },
    "Quantum Computing": {
        "etf": "QTUM",
        "holdings": [
            "MU", "2454.TW", "INTC", "STM", "ARM", "NOK", "AMD",
            "MRVL", "IFX.DE", "ON", "IONQ", "RGTI", "QUBT", "ARQQ",
            "QBTS", "FORM", "ACLS", "BRKS", "MKSI", "CEVA",
        ],
    },
    "Aerospace & Defense": {
        "etf": "ITA",
        "holdings": [
            "GE", "RTX", "BA", "RKLB", "HWM", "TDG", "GD", "LHX",
            "LMT", "NOC", "HII", "AXON", "HEI", "TXT", "SPR",
            "KTOS", "BWXT", "AVAV", "LDOS", "ERJ",
        ],
    },
    "Space": {
        "etf": "UFO",
        "holdings": [
            "RKLB", "PL", "VSAT", "MDA.TO", "LUNR", "IRDM", "SIRI",
            "ASTS", "GSAT", "SESGL", "MNTS", "BKSY", "RDW", "SPIR",
            "VORB", "ACHR", "ASTR", "SATL", "LLAP", "MAXR",
        ],
    },
    "Clean Energy": {
        "etf": "ICLN",
        "holdings": [
            "BE", "FSLR", "NXT", "ENPH", "600900.SS", "PLUG", "SEDG",
            "VWS.CO", "EQTL3.SA", "SUZLON.BO", "AES", "NEE", "ORA",
            "CWEN", "MEL.NZ", "EDPR.LS", "ERG.MI", "ORSTED.CO",
            "SCATC.OL", "CSIQ",
        ],
    },
    "Solar": {
        "etf": "TAN",
        "holdings": [
            "FSLR", "NXT", "ENPH", "ENLT.TA", "SEDG", "DORL.TA",
            "RUN", "NOFR.TA", "HASI", "SHLS", "ARRY", "CSIQ", "NOVA",
            "MAXN", "JKS", "DQ", "SPWR", "AMPS", "FLNC", "STEM",
        ],
    },
    "Homebuilders": {
        "etf": "XHB",
        "holdings": [
            "MOD", "OC", "MAS", "BLD", "WSM", "CARR", "CVCO", "TT",
            "DHI", "MTH", "MHK", "PHM", "BLDR", "LEN", "NVR", "TOL",
            "TMHC", "KBH", "FBIN", "AZEK",
        ],
    },
    "Infrastructure": {
        "etf": "PAVE",
        "holdings": [
            "PWR", "CSX", "ETN", "HWM", "TT", "UNP", "NUE", "ROK",
            "NSC", "URI", "EMR", "FAST", "MLM", "VMC", "JCI", "STLD",
            "GNRC", "MAS", "WMS", "RHI",
        ],
    },
    "Digital Infrastructure": {
        "etf": "SRVR",
        "holdings": [
            "EQIX", "DLR", "AMT", "IRM", "SBAC", "CCI", "CLNX.MC",
            "NXT.AX", "0788.HK", "IRDM", "UNIT", "LUMN", "QTS",
            "CONE", "REXT", "INXN", "CCOI", "SIFY", "CRWN.AX",
            "GDS",
        ],
    },
    "Water": {
        "etf": "PHO",
        "holdings": [
            "WAT", "ROP", "FERG", "ECL", "AWK", "IEX", "MLI", "CNM",
            "XYL", "VLTO", "WTS", "PNR", "MWA", "AQUA", "SNN",
            "WTRG", "CWT", "SJW", "MSEX", "YORW",
        ],
    },
    "Regional Banks": {
        "etf": "KRE",
        "holdings": [
            "UMBF", "EWBC", "VLY", "BPOP", "PNFP", "WAL", "ZION",
            "FLG", "ASB", "WTFC", "CFG", "KEY", "HBAN", "RF", "FITB",
            "MTB", "CMA", "ALLY", "FHN", "SBNY",
        ],
    },
    "Fintech": {
        "etf": "FINX",
        "holdings": [
            "HOOD", "XYZ", "PYPL", "COIN", "FISV", "ADYEN.AS", "CRCL",
            "INTU", "SOFI", "AFRM", "FIS", "GPN", "WEX", "BILL",
            "FOUR", "TOST", "LSPD.TO", "DLO", "PAGS", "STNE",
        ],
    },
    "E-Commerce": {
        "etf": "IBUY",
        "holdings": [
            "PTON", "SPOT", "BBBY", "AFRM", "W", "ETSY", "MSM",
            "EBAY", "TRIP", "BKNG", "AMZN", "CHWY", "FTCH", "OSTK",
            "CVNA", "GRPN", "REAL", "VROOM", "WISH", "CPNG",
        ],
    },
    "Social Media": {
        "etf": "SOCL",
        "holdings": [
            "RDDT", "035420.KS", "META", "0700.HK", "1024.HK", "GOOGL",
            "BIDU", "NTES", "PINS", "SPOT", "SNAP", "TWTR", "MTCH",
            "ZI", "WB", "BILI", "IQ", "TME", "DOYU", "YY",
        ],
    },
    "Electric Vehicles": {
        "etf": "DRIV",
        "holdings": [
            "INTC", "QCOM", "NVDA", "GOOGL", "6285.TW", "IFX.DE",
            "TSLA", "MSFT", "STMPA.PA", "NBIS", "GM", "F", "RIVN",
            "LCID", "NIO", "XPEV", "LI", "APTV", "BWA", "ALB",
        ],
    },
    "3D Printing": {
        "etf": "PRNT",
        "holdings": [
            "VELO", "XMTR", "DDD", "SIE.DE", "SNPS", "HPQ", "PTC",
            "RSW.L", "DSY.PA", "ADSK", "MKFG", "NNDM", "DM", "SSYS",
            "MTLS", "RKLB", "PRLB", "XONE", "SLM", "MTRN",
        ],
    },
    "Gaming & Esports": {
        "etf": "HERO",
        "holdings": [
            "EA", "TTWO", "NTES", "259960.KS", "3293.TWO", "U",
            "7974.T", "9766.T", "9684.T", "CDR.WA", "RBLX", "SE",
            "ATVI", "HUYA", "DOYU", "GMBL", "SKLZ", "PLTK",
            "DKNG", "PENN",
        ],
    },
    "Ark Innovation": {
        "etf": "ARKK",
        "holdings": [
            "TSLA", "AMD", "HOOD", "CRSP", "TEM", "CRCL", "SHOP",
            "ROKU", "COIN", "PLTR", "SQ", "DKNG", "U", "BEAM",
            "PATH", "TWLO", "ZM", "TDOC", "RBLX", "EXAS",
        ],
    },
    "Internet": {
        "etf": "FDN",
        "holdings": [
            "AMZN", "CSCO", "META", "NFLX", "GOOGL", "GOOG", "ANET",
            "CRM", "SNOW", "BKNG", "UBER", "SHOP", "DASH", "ABNB",
            "ZM", "PINS", "SNAP", "TWLO", "OKTA", "DDOG",
        ],
    },
    "Software": {
        "etf": "IGV",
        "holdings": [
            "ORCL", "MSFT", "PANW", "PLTR", "CRWD", "CRM", "APP",
            "NOW", "ADBE", "CDNS", "SNPS", "INTU", "WDAY", "TEAM",
            "HUBS", "DDOG", "ANSS", "ZS", "BILL", "TTD",
        ],
    },
    "Medtech": {
        "etf": "IHI",
        "holdings": [
            "ISRG", "ABT", "SYK", "EW", "IDXX", "BDX", "MDT", "GEHC",
            "DXCM", "RMD", "BAX", "HOLX", "TFX", "ALGN", "PODD",
            "STVN", "NVST", "HAE", "MASI", "INSP",
        ],
    },
    "Insurance": {
        "etf": "IAK",
        "holdings": [
            "CB", "PGR", "TRV", "ALL", "MET", "AFL", "AIG", "HIG",
            "PRU", "ACGL", "CINF", "RNR", "WRB", "GL", "L",
            "EG", "AIZ", "UNM", "LNC", "FAF",
        ],
    },
    "Transports": {
        "etf": "IYT",
        "holdings": [
            "UNP", "UBER", "FDX", "DAL", "UAL", "ODFL", "CSX", "UPS",
            "NSC", "XPO", "JBHT", "EXPD", "CHRW", "KNX", "SAIA",
            "LSTR", "SNDR", "LUV", "AAL", "ALK",
        ],
    },
    "Airlines": {
        "etf": "JETS",
        "holdings": [
            "DAL", "AAL", "UAL", "LUV", "ULCC", "AC.TO", "JBLU",
            "ALGT", "ALK", "SKYW", "SAVE", "MESA", "HA", "RYAAY",
            "CPA", "GOL", "AZUL", "WJA.TO", "CEA", "ZNH",
        ],
    },
    "Food & Bev": {
        "etf": "PBJ",
        "holdings": [
            "ADM", "MNST", "CTVA", "MDLZ", "KR", "PEP", "SYY", "HSY",
            "UNFI", "DAR", "KO", "GIS", "KHC", "CAG", "HRL", "CPB",
            "BG", "INGR", "POST", "LNDC",
        ],
    },
    "Agriculture": {
        "etf": "MOO",
        "holdings": [
            "BAYN.DE", "DE", "CTVA", "ADM", "NTR.TO", "TSN", "ZTS",
            "CF", "6326.T", "BG", "FMC", "MOS", "ANDE", "AGCO",
            "CNHI", "IPI", "AVD", "SMG", "LMNR", "CALM",
        ],
    },
    "Retail": {
        "etf": "XRT",
        "holdings": [
            "GO", "SAH", "ETSY", "AMZN", "M", "BBY", "VSXY", "EBAY",
            "CVNA", "HZO", "TGT", "DG", "DLTR", "FIVE", "ROST",
            "BURL", "ULTA", "GPS", "ANF", "AEO",
        ],
    },
    "Leisure & Entertainment": {
        "etf": "PEJ",
        "holdings": [
            "EXPE", "MAR", "ABNB", "UAL", "LION", "LVS", "RCL", "CCL",
            "SYY", "VSNT", "HLT", "WYNN", "MGM", "NCLH", "DIS",
            "DKNG", "MTN", "SIX", "FUN", "SEAS",
        ],
    },
    "Rare Earth & Critical Minerals": {
        "etf": "REMX",
        "holdings": [
            "PLS.AX", "ALB", "LYC.AX", "LTR.AX", "600111.SS", "MP",
            "SQM", "601958.SS", "01772", "600549.SS", "LTHM", "LAC",
            "SGML", "UUUU", "AMAM", "HRE.AX", "ARU.AX", "NB",
            "QMC.V", "VML.AX",
        ],
    },
    "MLPs & Pipelines": {
        "etf": "AMLP",
        "holdings": [
            "PAA", "WES", "SUN", "ET", "EPD", "MPLX", "HESM", "CQP",
            "USAC", "GEL", "DCP", "TRGP", "OKE", "KMI", "WMB",
            "ENLC", "AM", "CEQP", "CPLP", "KNOP",
        ],
    },
    "IPO & New Listings": {
        "etf": "IPO",
        "holdings": [
            "ARM", "ALAB", "CRWV", "KVUE", "RDDT", "VIK", "CRCL",
            "MDLN", "RBRK", "AHR", "CART", "BIRK", "KOKN", "LRN",
            "TOST", "ONON", "CLBT", "GFS", "CAVA", "DV",
        ],
    },
    "Mortgage REITs": {
        "etf": "REM",
        "holdings": [
            "NLY", "AGNC", "STWD", "DX", "RITM", "BXMT", "ARR",
            "EFC", "ARI", "TWO", "PMT", "MFA", "NYMT", "GPMT",
            "KREF", "CIM", "IVR", "RC", "ABR", "MITT",
        ],
    },
    "Shipping": {
        "etf": "BOAT",
        "holdings": [
            "FRO", "MATX", "1308.HK", "9104.T", "MAERSK-B.CO",
            "9107.T", "STNG", "HAFNI.OL", "0316.HK", "WAWI.OL",
            "ZIM", "DAC", "GOGL", "SBLK", "GNK", "INSW", "TNK",
            "CMRE", "FLNG", "EURN",
        ],
    },
    "Timber & Forestry": {
        "etf": "WOOD",
        "holdings": [
            "WY", "WFG.TO", "SUZB3.SA", "SW", "IP", "KLBN11",
            "STERV.HE", "SLVM", "UPM.HE", "SCA B", "RYN", "PCH",
            "CTT", "MLP.PA", "MERC", "POPE", "LPX", "BCC",
            "UFPI", "CSAN3.SA",
        ],
    },

    # ── Commodities ──

    "Gold": {
        "etf": "GLD",
        # Physical gold has no equity; use gold MINERS so the sleeve picks a cheap producer (value)
        # instead of bullion. Pick logic filters to those with candles + positive P/B.
        "holdings": [
            "NEM", "GOLD", "AEM", "KGC", "FNV", "WPM", "RGLD", "AU", "GFI",
            "BTG", "EGO", "IAG", "NGD", "SSRM", "HMY", "OR", "HL", "PAAS",
        ],
    },
    "Silver": {
        "etf": "SLV",
        # Physical silver -> silver MINERS.
        "holdings": [
            "CDE", "HL", "PAAS", "AG", "MAG", "EXK", "SSRM", "FSM", "SVM",
            "GATO", "SILV", "GORO", "USAS", "MUX",
        ],
    },
    "Platinum": {
        "etf": "PPLT",
        "holdings": [],  # Holds physical platinum
    },
    "Oil": {
        "etf": "USO",
        # Oil futures -> oil & gas PRODUCERS / E&P so the sleeve picks a cheap producer.
        "holdings": [
            "EOG", "DVN", "FANG", "OXY", "CTRA", "AR", "COP", "XOM", "CVX",
            "HES", "APA", "MRO", "OVV", "MTDR", "PR", "CHRD", "SM", "CIVI",
        ],
    },
    "Natural Gas": {
        "etf": "UNG",
        "holdings": [],  # Holds natural gas futures
    },
    "Agriculture Commodities": {
        "etf": "DBA",
        "holdings": [],  # Holds agriculture futures (corn, soybeans, wheat, sugar, etc.)
    },
    "Wheat": {
        "etf": "WEAT",
        "holdings": [],  # Holds wheat futures
    },
    "Corn": {
        "etf": "CORN",
        "holdings": [],  # Holds corn futures
    },
    "Uranium": {
        "etf": "URA",
        "holdings": [
            "CCO.TO", "OKLO", "NXE.TO", "UEC", "KAP", "EFR.TO",
            "PDN.AX", "047040.KS", "028260.KS", "LEU", "DNN", "URG",
            "UUUU", "SMR", "BWXT", "GEL.AX", "BMN.AX", "FSY.TO",
            "HAM.AX", "AGE.AX",
        ],
    },
    "Lithium & Battery": {
        "etf": "LIT",
        "holdings": [
            "RIO", "6762.T", "ALB", "006400.KS", "002371.SZ", "6752.T",
            "TSLA", "373220.KS", "PLS.AX", "300750.SZ", "SQM", "LTHM",
            "LAC", "SGML", "LIVENT", "QS", "MVST", "ENVX", "SLDP",
            "FREYR",
        ],
    },
    "Copper Miners": {
        "etf": "COPX",
        "holdings": [
            "HBM.TO", "TECK-B.TO", "BHP.AX", "ANTO.L", "FM.TO",
            "KGH.WA", "BOL.ST", "SCCO", "GLEN.L", "LUN.TO", "FCX",
            "IVPAF", "ERO.TO", "TRQ", "CS.TO", "FILO.TO", "OZL.AX",
            "SFR.AX", "SOLG.ST", "CU.TO",
        ],
    },
    "Steel": {
        "etf": "SLX",
        "holdings": [
            "BHP", "RIO", "NUE", "RIO.AX", "VALE", "STLD", "FMG.AX",
            "MT", "RS", "PKX", "CLF", "X", "AA", "CMC", "GGB",
            "SID", "TX", "SCHN", "WOR", "ZEUS",
        ],
    },

    # ── Fixed Income ──

    "Bonds (20Y Treasury)": {
        "etf": "TLT",
        "holdings": [],  # Holds US Treasury bonds
    },
    "Bonds (Agg)": {
        "etf": "AGG",
        "holdings": [],  # Holds aggregate bond index
    },
    "High Yield Bonds": {
        "etf": "HYG",
        "holdings": [],  # Holds high yield corporate bonds
    },
    "TIPS (Inflation)": {
        "etf": "TIP",
        "holdings": [],  # Holds Treasury Inflation-Protected Securities
    },

    # ── Factor / Style ──

    "Dividend": {
        "etf": "SCHD",
        "holdings": [
            "QCOM", "TXN", "UNH", "KO", "MRK", "CVX", "VZ", "PG",
            "COP", "AMGN", "ABBV", "PFE", "CSCO", "HD", "PEP",
            "BLK", "USB", "LMT", "TGT", "AVGO",
        ],
    },
    "Momentum": {
        "etf": "MTUM",
        "holdings": [
            "MU", "AMD", "AVGO", "INTC", "CAT", "XOM", "LRCX", "JNJ",
            "GEV", "GOOGL", "NVDA", "LLY", "PLTR", "GE", "APP",
            "ANET", "KLAC", "VST", "CEG", "TRGP",
        ],
    },
    "Value": {
        "etf": "VTV",
        "holdings": [
            "MU", "JPM", "BRK-B", "XOM", "JNJ", "WMT", "INTC",
            "CSCO", "CAT", "ABBV", "BAC", "CVX", "PG", "KO", "MRK",
            "PFE", "VZ", "T", "COP", "GS",
        ],
    },
    "Growth": {
        "etf": "VUG",
        "holdings": [
            "NVDA", "AAPL", "MSFT", "GOOGL", "AVGO", "AMZN", "GOOG",
            "META", "TSLA", "LLY", "CRM", "NOW", "ADBE", "AMD",
            "NFLX", "PLTR", "INTU", "ISRG", "SNPS", "CDNS",
        ],
    },
    "Low Volatility": {
        "etf": "SPLV",
        "holdings": [
            "FE", "L", "BRK-B", "DUK", "AEE", "WEC", "LNT", "CNP",
            "DTE", "EVRG", "ED", "XEL", "ES", "PPL", "AEP", "SO",
            "CMS", "PEG", "ATO", "NI",
        ],
    },

    # ── Crypto ETFs (stock-based) ──

    "Bitcoin ETF": {
        "etf": "IBIT",
        "holdings": [],  # Holds Bitcoin
    },
    "Ethereum ETF": {
        "etf": "ETHA",
        "holdings": [],  # Holds Ethereum
    },

    # ── Crypto (actual) ──

    "Bitcoin": {
        "etf": "BTC-USD",
        "holdings": [],
    },

    # ── International ──

    "Emerging Markets": {
        "etf": "EEM",
        "holdings": [
            "2330.TW", "005930.KS", "000660.KS", "0700.HK", "9988.HK",
            "2454.TW", "2308.TW", "2317.TW", "005935", "00939",
            "BABA", "PDD", "JD", "VALE", "PBR", "ITUB", "NU",
            "INFY", "HDB", "WIT",
        ],
    },
    "International (EAFE)": {
        "etf": "EFA",
        "holdings": [
            "ASML.AS", "HSBA.L", "ROP.SW", "AZN.L", "NOVN.SW",
            "NESN.SW", "SHEL.L", "SIE.DE", "BHP.AX", "8306.T",
            "SAP.DE", "MC.PA", "OR.PA", "RMS.PA", "ULVR.L", "BP.L",
            "GSK.L", "RIO.L", "TTE.PA", "DGE.L",
        ],
    },
    "China": {
        "etf": "MCHI",
        "holdings": [
            "0700.HK", "9988.HK", "00939", "1810.HK", "01398",
            "02318", "PDD", "3690.HK", "03988", "9999.HK",
            "BABA", "JD", "BIDU", "NIO", "LI", "XPEV", "NTES",
            "TCOM", "ZTO", "BILI",
        ],
    },
    "India": {
        "etf": "INDA",
        "holdings": [
            "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "BHARTIARTL.NS", "INFY.NS", "AXISBANK.BO", "LT.NS",
            "M&M.NS", "TCS.NS", "HCLTECH.NS", "SBIN.NS",
            "BAJFINANCE.NS", "TITAN.NS", "MARUTI.NS", "TATAMOTORS.NS",
            "NESTLEIND.NS", "ITC.NS", "WIPRO.NS", "SUNPHARMA.NS",
            "ADANIENT.NS",
        ],
    },
    "Japan": {
        "etf": "EWJ",
        "holdings": [
            "8306.T", "7203.T", "9984.T", "8035.T", "6501.T",
            "8316.T", "285A.T", "6758.T", "6857.T", "8411.T",
            "6861.T", "6954.T", "7974.T", "4063.T", "9432.T",
            "6098.T", "8058.T", "4502.T", "6367.T", "9433.T",
        ],
    },
    "South Korea": {
        "etf": "EWY",
        "holdings": [
            "000660.KS", "005930.KS", "009150.KS", "402340.KS",
            "005380.KS", "105560.KS", "012330.KS", "034020.KS",
            "006400.KS", "000270.KS", "035420.KS", "259960.KS",
            "068270.KS", "055550.KS", "028260.KS", "003550.KS",
            "047040.KS", "207940.KS", "373220.KS", "066570.KS",
        ],
    },
    "Latin America": {
        "etf": "ILF",
        "holdings": [
            "VALE", "NU", "ITUB", "GMEXICOB.MX", "PBR",
            "GFNORTEO.MX", "BAP", "AMXB.MX", "FEMSAUBD.MX", "BBD",
            "ABEV", "SQM", "BSBR", "CSAN3.SA", "WEGE3.SA",
            "GGAL", "CRFB3.SA", "RENT3.SA", "SUZB3.SA", "LREN3.SA",
        ],
    },
    "Brazil": {
        "etf": "EWZ",
        "holdings": [
            "VALE3.SA", "NU", "ITUB4", "PETR4", "PETR3.SA", "BBDC4",
            "B3SA3.SA", "ABEV3.SA", "WEGE3.SA", "SBSP3.SA", "RENT3.SA",
            "SUZB3.SA", "BBAS3.SA", "ELET3.SA", "CSAN3.SA", "LREN3.SA",
            "CRFB3.SA", "RADL3.SA", "TOTS3.SA", "HAPV3.SA",
        ],
    },
    "Mexico": {
        "etf": "EWW",
        "holdings": [
            "GMEXICOB.MX", "GFNORTEO.MX", "AMXB.MX", "FEMSAUBD.MX",
            "CEMEXCPO.MX", "PE&OLES.MX", "WALMEX.MX", "GAPB.MX",
            "AC.MX", "ASURB.MX", "BIMBOA.MX", "LABB.MX",
            "ELEKTRA.MX", "OMAB.MX", "KOFUBL.MX", "GCARSOA1.MX",
            "GRUMAB.MX", "ALSEA.MX", "SITES1A-1.MX", "PINFRA.MX",
        ],
    },
    "Europe": {
        "etf": "VGK",
        "holdings": [
            "ASML.AS", "HSBA.L", "ROP.SW", "NOVN.SW", "AZN.L",
            "NESN.SW", "SHEL.L", "SIE.DE", "SAP.DE", "SAN.MC",
            "MC.PA", "OR.PA", "TTE.PA", "ALV.DE", "BNP.PA",
            "DGE.L", "BP.L", "GSK.L", "RIO.L", "ULVR.L",
        ],
    },
    "Germany": {
        "etf": "EWG",
        "holdings": [
            "SIE.DE", "SAP.DE", "ALV.DE", "ENR.DE", "IFX.DE",
            "DTE.DE", "RHM.DE", "MUV2.DE", "DBK.DE", "DHL.DE",
            "BAS.DE", "BMW.DE", "MBG.DE", "ADS.DE", "HEN3.DE",
            "FRE.DE", "VOW3.DE", "SY1.DE", "BEI.DE", "LIN.DE",
        ],
    },
    "UK": {
        "etf": "EWU",
        "holdings": [
            "HSBA.L", "AZN.L", "SHEL.L", "RR.L", "BATS.L", "ULVR.L",
            "RIO.L", "BP.L", "GSK.L", "BARC.L", "LSEG.L", "DGE.L",
            "REL.L", "NG.L", "LLOY.L", "ABF.L", "AAL.L", "VOD.L",
            "EXPN.L", "ANTO.L",
        ],
    },
    "Australia": {
        "etf": "EWA",
        "holdings": [
            "BHP.AX", "CBA.AX", "WBC.AX", "NAB.AX", "ANZ.AX",
            "WES.AX", "MQG.AX", "RIO.AX", "GMG.AX", "WDS.AX",
            "CSL.AX", "TLS.AX", "WOW.AX", "ALL.AX", "COL.AX",
            "FMG.AX", "STO.AX", "TCL.AX", "REA.AX", "JHX.AX",
        ],
    },
    "Africa": {
        "etf": "AFK",
        "holdings": [
            "ATW", "AAL.L", "EDV.L", "GTCO", "AAF.L", "ZENITH",
            "NPN.JO", "IVN.TO", "BCP", "FM.TO", "SBK.JO", "FSR.JO",
            "AGL.JO", "SOL.JO", "MTN.JO", "VOD.JO", "NED.JO",
            "AMS.JO", "SHP.JO", "BID.JO",
        ],
    },

    # ── Size ──

    "Small Cap": {
        "etf": "IWM",
        "holdings": [
            "BE", "CRDO", "STRL", "IONQ", "FN", "NXT", "CDE", "TTMI",
            "ECHO", "GH", "SMCI", "AEHR", "LNTH", "AMBA", "SWX",
            "KTOS", "UFPI", "POWI", "CALM", "RBC",
        ],
    },
    "Micro Cap": {
        "etf": "IWC",
        "holdings": [
            "VERX", "CORT", "EXTR", "CALX", "PRFT", "TBBK", "STEP",
            "CRVL", "ARCB", "PLMR", "RAMP", "KNTK", "WRBY", "HQY",
            "ALRM", "IIPR", "DVAX", "TMDX", "PCVX", "GPOR",
        ],
    },
    "Nanotechnology": {
        "etf": "TINY",
        "holdings": [
            "BRKR", "6082.HK", "LRCX", "1963.T", "CBT", "VECO",
            "ASML.AS", "7731.T", "2330.TW", "Q", "ACLS", "IPGP",
            "CEVA", "FORM", "ONTO", "POWI", "UCTT", "ENTG",
            "OLED", "LSCC",
        ],
    },
    "Nasdaq 100": {
        "etf": "QQQ",
        "holdings": [
            "NVDA", "AAPL", "MSFT", "MU", "AMZN", "AMD", "GOOGL",
            "TSLA", "AVGO", "GOOG", "META", "NFLX", "COST", "INTC",
            "QCOM", "CRM", "NOW", "ADBE", "TXN", "INTU",
        ],
    },

    # ── Alts ──

    "Blockchain Stocks": {
        "etf": "BLOK",
        "holdings": [
            "HUT", "CIFR", "WULF", "GLXY", "CORZ", "DELL", "HOOD",
            "APLD", "MARA", "RIOT", "COIN", "MSTR", "CLSK", "SI",
            "BITF", "BTBT", "ARBK", "IREN", "HIVE", "SOS",
        ],
    },
    "Cannabis": {
        "etf": "MSOS",
        "holdings": [
            "CURA", "GTII", "TRSSF", "TCNNF", "VRNOF", "CRLBF",
            "AYRWF", "JUSHF", "MRMD", "GRNH", "CCHWF", "CURLF",
            "GLASF", "TLRY", "CGC", "ACB", "SNDL", "OGI",
            "HEXO", "VFF",
        ],
    },
}


def get_holdings(sector_name: str) -> list[str]:
    """Get the holdings ticker list for a sector."""
    data = HOLDINGS.get(sector_name)
    if data:
        return data["holdings"]
    return []


def get_all_unique_tickers() -> list[str]:
    """Get all unique stock tickers across all sectors."""
    seen = set()
    tickers = []
    for data in HOLDINGS.values():
        for t in data["holdings"]:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    return tickers


def get_sectors_for_ticker(ticker: str) -> list[str]:
    """Find which sectors a ticker appears in."""
    result = []
    for sector_name, data in HOLDINGS.items():
        if ticker in data["holdings"]:
            result.append(sector_name)
    return result
