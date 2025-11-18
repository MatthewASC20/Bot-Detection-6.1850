# YouTube Botnet Detection System - Setup Guide

## Overview
This proof of concept system detects coordinated bot networks on YouTube by analyzing comment patterns, temporal behaviors, and network relationships.

## Prerequisites
- Python 3.8 or higher
- YouTube Data API v3 credentials
- At least 4GB of RAM for processing thousands of comments

## Installation Steps

### 1. Clone/Download the Project
```bash
# Create project directory
mkdir youtube_botnet_detector
cd youtube_botnet_detector
```

### 2. Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up YouTube Data API

#### Getting API Credentials:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable YouTube Data API v3:
   - Go to "APIs & Services" > "Library"
   - Search for "YouTube Data API v3"
   - Click on it and press "Enable"
4. Create credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy the API key

#### Configure API Key:
Create a `.env` file in the project root:
```
YOUTUBE_API_KEY=your_api_key_here
# Optional: Multiple API keys for rotation (to handle quotas)
YOUTUBE_API_KEY_2=backup_api_key
YOUTUBE_API_KEY_3=another_backup_key
```

### 4. Download NLTK Data (for text analysis)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

## Project Structure
```
youtube_botnet_detector/
│
├── config/
│   └── config.py          # Configuration settings
│
├── data_collection/
│   ├── __init__.py
│   ├── youtube_api.py     # YouTube API wrapper
│   └── data_collector.py  # Main data collection module
│
├── feature_extraction/
│   ├── __init__.py
│   ├── temporal_features.py   # Time-based pattern detection
│   ├── text_features.py       # Comment similarity & NLP
│   ├── network_features.py    # User interaction patterns
│   └── behavioral_features.py # Account behavior analysis
│
├── detection/
│   ├── __init__.py
│   ├── clustering.py      # Unsupervised clustering algorithms
│   ├── supervised.py      # Supervised classification (optional)
│   └── ensemble.py        # Combined detection methods
│
├── visualization/
│   ├── __init__.py
│   ├── network_viz.py     # Network graph generation
│   └── dashboard.py       # Interactive visualizations
│
├── storage/
│   ├── __init__.py
│   ├── database.py        # SQLite database handler
│   └── models.py          # Data models
│
├── utils/
│   ├── __init__.py
│   ├── rate_limiter.py   # API rate limiting
│   └── helpers.py         # Utility functions
│
├── data/
│   ├── raw/              # Raw API responses
│   ├── processed/        # Processed features
│   ├── labeled/          # Your labeled bot data
│   └── results/          # Detection results
│
├── outputs/
│   ├── graphs/           # Network visualizations
│   └── reports/          # Analysis reports
│
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
├── .env                 # API keys (create this)
└── README.md            # This file
```

## Usage

### Basic Usage - Analyze Arbitrary URLs:
```python
python main.py --urls "video_url1,video_url2,video_url3" --max-comments 1000
```

### Analyze Political Channels:
```python
python main.py --mode political --max-comments 5000
```

### Use Labeled Data for Semi-Supervised Learning:
```python
python main.py --urls "video_urls" --labeled-data "data/labeled/bot_comments.csv"
```

### Command Line Arguments:
- `--urls`: Comma-separated YouTube video URLs
- `--mode`: Predefined modes ('political', 'custom')
- `--max-comments`: Maximum comments per video (default: 1000)
- `--labeled-data`: Path to CSV with labeled bot comments
- `--output-dir`: Directory for results (default: outputs/)
- `--visualize`: Generate network graphs (default: True)

## API Quota Management
YouTube Data API has the following quotas:
- **Free tier**: 10,000 units per day
- **Comment threads list**: 1 unit per request (returns up to 100 comments)
- **Channel list**: 1 unit per request

To work with thousands of comments:
- The system implements caching to avoid redundant API calls
- Use multiple API keys (configured in .env) for automatic rotation
- Consider spreading data collection over multiple days for large datasets

## Detection Methods

The system uses multiple detection strategies:

1. **Temporal Clustering**: Identifies accounts posting in coordinated time windows
2. **Text Similarity**: Detects template-based or copied comments
3. **Network Analysis**: Finds groups of accounts that frequently appear together
4. **Behavioral Patterns**: Analyzes account age, posting frequency, username patterns
5. **Ensemble Detection**: Combines all methods for final bot probability score

## Output

The system generates:
- **Network graphs** showing bot clusters (in `outputs/graphs/`)
- **CSV reports** with bot probability scores (in `outputs/reports/`)
- **SQLite database** with all collected data (in `data/botnet_detection.db`)
- **Interactive HTML dashboard** for exploration

## Ethical Considerations
This tool is for educational and research purposes only. Please:
- Respect YouTube's Terms of Service
- Do not use for harassment or targeting individuals
- Consider privacy implications when sharing results
- Use findings responsibly and verify before taking action

## Troubleshooting

### API Key Issues:
- Ensure API key is correctly set in .env file
- Check that YouTube Data API v3 is enabled in Google Cloud Console
- Verify billing is enabled if using paid quota

### Memory Issues:
- Reduce `--max-comments` parameter
- Process videos in batches
- Increase system swap space if needed

### Rate Limiting:
- The system automatically handles rate limits
- Add more API keys to .env for faster collection
- Use cached data when re-running analysis

## Next Steps
After initial setup:
1. Run a small test with 2-3 videos to verify setup
2. Gradually increase scope to thousands of comments
3. Fine-tune detection thresholds based on results
4. Integrate your labeled dataset for improved accuracy
