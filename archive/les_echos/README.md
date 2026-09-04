# CAC 40 News Scraper

**Extract French financial news articles mentioning CAC 40 companies from Les Echos (2010-2018)**

## Overview

This project automatically extracts **financial news articles** from **Les Echos**, France's leading financial newspaper. The spider identifies and collects all news mentioning **CAC 40 companies**, then exports articles to CSV and JSON formats.

## Features

✅ **CAC 40 Focus** - Tracks all 40 major French companies  
✅ **Smart Company Detection** - Automatically identifies companies mentioned  
✅ **Dual Export Formats** - CSV and JSON outputs  
✅ **Duplicate Prevention** - Automatically removes duplicate articles  
✅ **Rate Limiting** - Respects website limits with smart throttling  
✅ **Easy Deployment** - Simple Python + Scrapy setup  

## Quick Start

### 1. Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/cac40-news-scraper.git
cd cac40-news-scraper

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run News Scraper

```bash
# Extract all CAC 40 news
scrapy crawl les_echos_news

# Or use the utility script
python run_all_spiders.py
```

### 3. Access Results

Check `data/` directory:
- `news_articles.csv` - Spreadsheet format
- `news_articles.json` - JSON format

## Output Format

### CSV Example
```
headline,summary,publish_date,companies_mentioned,categories,author,source_url,extract_date
"Total dépasse les attentes",...,"2023-01-15","TOTAL;Shell",...,Sophie Martin,...
```

### JSON Example
```json
{
  "headline": "Total dépasse les attentes",
  "summary": "Le géant de l'énergie rapporte...",
  "publish_date": "2023-01-15",
  "companies_mentioned": ["TotalEnergies", "TOTAL"],
  "categories": ["Finance", "Énergie"],
  "author": "Sophie Martin",
  "source_url": "https://lesechos.fr/...",
  "extract_date": "2024-01-15T10:30:45"
}
```

## Data Extracted

Each news article includes:

| Field | Description |
|-------|-------------|
| **headline** | Article title |
| **summary** | Short description |
| **content** | Full article text |
| **publish_date** | Publication date |
| **companies_mentioned** | CAC 40 companies mentioned |
| **categories** | Article categories (Finance, Énergie, etc.) |
| **author** | Article author |
| **source_url** | Original article link |
| **keywords** | Extracted keywords |
| **extract_date** | When data was scraped |

## Configuration

Edit `cac40_scraper/settings.py`:

```python
# Rate limiting
DOWNLOAD_DELAY = 2              # Delay between requests (seconds)
CONCURRENT_REQUESTS = 4         # Max concurrent requests

# Logging
LOG_LEVEL = 'INFO'              # Options: DEBUG, INFO, WARNING, ERROR

# HTTP caching
HTTPCACHE_ENABLED = True        # Enable for faster re-runs
```

## CAC 40 Companies Tracked

Accor • L'Oréal • Hermès • LVMH • Pernod Ricard • TotalEnergies • Safran • EDF • Alstom • Crédit Agricole • AXA • BNP Paribas • Société Générale • Air France-KLM • Sanofi • EssilorLuxottica • Vivendi • Publicis • Atos • Vinci • Capgemini • Legrand • Michelin • Sodexo • Saint-Gobain • Orange • Renault • Stellantis • Teleperformance • Unilever • Airbus • Altarea Cogedim • Arkema • Bouygues • Dassault Systèmes • Getlink • ICADE • Koninklijke Philips • Schneider Electric • STMicroelectronics

## Common Commands

```bash
# Run the news spider
scrapy crawl les_echos_news

# View logs in real-time
tail -f logs/scraper.log

# Debug with interactive shell
scrapy shell https://www.lesechos.fr/bourse/

# Set debug level
set LOG_LEVEL=DEBUG
scrapy crawl les_echos_news
```

## Project Structure

```
cac40-news-scraper/
├── cac40_scraper/
│   ├── spiders/
│   │   └── les_echos_news.py      # News extraction spider
│   ├── items/
│   │   └── financial_items.py     # NewsItem definition
│   ├── pipelines/
│   │   └── data_pipelines.py      # CSV/JSON export
│   └── settings.py                # Scrapy configuration
├── data/                          # Output files (auto-created)
├── logs/                          # Log files (auto-created)
├── requirements.txt               # Python dependencies
└── run_all_spiders.py            # Utility script
```

## Requirements

- Python 3.8+
- Scrapy 2.11.0
- lxml 4.9.3
- See `requirements.txt` for complete list

## Limitations & Considerations

⚠️ Website structure changes may affect selectors  
⚠️ Historical data (2010-2018) may have limited availability  
⚠️ Rate limiting enforced to be respectful to Les Echos  
⚠️ Review website's Terms of Service before scraping  

## License

MIT License - See [LICENSE](LICENSE) file

## Support

- 📖 See [SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md) for troubleshooting
- 📝 See [QUICKSTART.md](QUICKSTART.md) for quick reference
- 🤝 See [CONTRIBUTING.md](docs/CONTRIBUTING.md) to contribute

## Usage Rights

✅ Intended for research and educational purposes  
✅ Respects robots.txt and rate limits  
✅ Provides proper User-Agent identification  
⚠️ Check Les Echos Terms of Service  

---

**Ready to extract news?** → `scrapy crawl les_echos_news`
