# YouTube Botnet Detection System

Educational toolkit for collecting YouTube comments, extracting 80+ behavioral/text/temporal/network/semantic features, and detecting coordinated botnets. Two detection paths are available:
- **BotBuster (default):** semantic + temporal coordination with cross-video analysis and comment-level probabilities.
- **Original clustering:** feature-based ensemble using HDBSCAN/DBSCAN on the full feature set.

## What’s included
- YouTube Data API wrapper with key rotation, caching, and channel/video enrichment.
- Feature extractors: temporal, text/NLP, network co-occurrence + replies, behavioral profiling, semantic similarity (OpenAI or TF-IDF/ topic fallback).
- BotBuster coordination detector, legacy clustering detector, visualizations, and SQLite storage.
- Discovery helpers for expanding suspected botnets and an integrated Flask API for the Chrome extension.
- Prebuilt visual reports (`outputs/graphs`, `outputs/reports`) and comparison tooling.

## Requirements
- Python 3.9+ recommended.
- YouTube Data API v3 key in `.env` (required; multiple keys supported for rotation).
- Optional: `OPENAI_API_KEY` for higher-quality semantic features (falls back to local TF-IDF/intent/topic features).
- NLTK data: `punkt`, `stopwords`, `vader_lexicon`.

## Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Download NLTK data for text features
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

Create `.env` in the repo root:
```
YOUTUBE_API_KEY=your_key_here
YOUTUBE_API_KEY_2=optional_second_key
YOUTUBE_API_KEY_3=optional_third_key
OPENAI_API_KEY=optional_semantic_key
```

## Running detection (CLI)
BotBuster (semantic + temporal coordination, default):
```bash
python main.py --mode urls --urls "https://youtube.com/watch?v=VIDEO1,https://youtube.com/watch?v=VIDEO2" --max-comments 800 --method botbuster
```

Original feature-based clustering:
```bash
python main.py --mode political --max-comments 500 --method original
```

Key flags:
- `--mode {urls,political,search}`: collection strategy.
- `--urls`: comma-separated video URLs/IDs (urls mode).
- `--max-comments`: per-video cap (default 1000).
- `--labeled-data`: CSV with `text,is_bot` for semi-supervised context.
- `--method {botbuster,original,both}`: choose detection path; `both` runs both and reports separately.
- `--no-cache`: disable API response cache; `--clear-db`: reset SQLite before run.

Outputs live in `outputs/graphs` (HTML visualizations) and `outputs/reports` (CSV/JSON summaries); comments and profiles are stored in `data/botnet_detection.db`.

### Output viewer (local UI)
Browse generated HTML reports without hunting through folders:
```bash
python ui/report_viewer.py --port 5050
```
Then open `http://127.0.0.1:5050` to see a sidebar list of recent graph/report HTML files and a live preview pane. Use the refresh button to pick up new runs; “Open in new tab” lets you view a file directly.

## Other entry points
- **Comparison run:** `python run_comparison.py --mode urls --urls "VIDEO_URLS"` to benchmark BotBuster vs. original clustering.
- **Extension/Batch API:** `python api/integrated_api.py` (Flask on `:5001`) powers the Chrome extension and batch `/api/analyze_video` endpoint, using the same feature pipeline.
- **Discovery helpers:** `discovery/` contains expansion + identification utilities for surfacing related botnets from seed accounts/videos.

## Repository map
```
config/                # Global settings and directory setup
data_collection/       # YouTube API wrapper + orchestrated collectors
features/              # temporal, text, network, behavioral, semantic feature extractors
detection/             # BotBuster coordination detector + clustering detector
visualization/         # Plotly/Matplotlib HTML reports
storage/               # SQLite handler and schema bootstrap
api/                   # Flask API for extension/batch analysis
discovery/             # Botnet expansion & identification helpers
youtube-bot-detector-extension/  # Chrome extension + docs
```

## Troubleshooting
- **Missing API keys:** `.env` must include at least one `YOUTUBE_API_KEY`; Config raises if absent.
- **NLTK errors:** rerun the download command in Setup.
- **Quota limits:** add backup keys in `.env` for automatic rotation, or re-run with caching enabled.
- **Large datasets:** lower `--max-comments`, or process in batches to keep memory manageable.

Use responsibly and in accordance with YouTube’s Terms of Service. This project is for research and educational purposes only.
