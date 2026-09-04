# Quick Start - CAC 40 News Scraper

## 3-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run News Spider
```bash
scrapy crawl les_echos_news
```

### 3. Check Results
```bash
dir data\  # Windows
ls data/   # Mac/Linux
```

You'll see:
- `news_articles.csv` - All articles in spreadsheet format
- `news_articles.json` - All articles in JSON format

---

## Common Commands

```bash
# Extract CAC 40 news
scrapy crawl les_echos_news

# View extraction logs
type logs\scraper.log  # Windows
tail logs/scraper.log  # Unix

# Debug a specific URL
scrapy shell https://www.lesechos.fr/finance-marches/

# Interactive debugging
>>> response.css('article::text').get()
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "scrapy: command not found" | Activate venv: `.\.venv\Scripts\activate` |
| "ModuleNotFoundError" | Reinstall: `pip install -r requirements.txt` |
| "SSL certificate error" | See [SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md) |
| No output files | Check `logs/scraper.log` for errors |

---

## Output Examples

### news_articles.csv
```
headline | publish_date | companies_mentioned | source_url
---------|-----|------|------
TotalEnergies reports earnings | 2023-01-15 | TOTAL;TotalEnergies | https://...
LVMH expands portfolio | 2023-01-16 | MC;LVMH | https://...
```

### news_articles.json
```json
[
  {
    "headline": "TotalEnergies reports earnings",
    "publish_date": "2023-01-15",
    "companies_mentioned": ["TotalEnergies", "TOTAL"],
    "extract_date": "2024-01-15T10:30:45"
  }
]
```

---

## Next Steps

1. 📖 Read [README.md](README.md) for full documentation
2. 🛠️ Setup details: [SETUP_INSTRUCTIONS.md](docs/SETUP_INSTRUCTIONS.md)
3. 🤝 Contribute: [CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

**Ready?** → `scrapy crawl les_echos_news`
