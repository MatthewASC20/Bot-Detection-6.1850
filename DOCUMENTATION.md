# YouTube Botnet Detection System - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Detection Methods](#detection-methods)
4. [Features Extracted](#features-extracted)
5. [API Usage Examples](#api-usage-examples)
6. [Customization Guide](#customization-guide)
7. [Understanding Results](#understanding-results)

## Overview

This proof-of-concept system detects coordinated bot networks on YouTube by analyzing multiple behavioral signals:
- **Temporal patterns**: Burst posting, synchronized timing, regular intervals
- **Text similarity**: Template detection, spam patterns, low diversity
- **Network analysis**: Co-occurrence patterns, community detection, clique formation
- **Behavioral indicators**: Account age, username patterns, targeting behavior

## Architecture

### Modular Design
```
Data Collection → Feature Extraction → Detection → Visualization
      ↓                   ↓                ↓            ↓
YouTube API      Temporal, Text,    Clustering    Network Graphs
                Network, Behavioral  Algorithms    & Reports
```

### Key Components

1. **Data Collection** (`data_collection/`)
   - `youtube_api.py`: YouTube Data API v3 wrapper with rate limiting
   - `data_collector.py`: Orchestrates data gathering from videos/channels

2. **Feature Extraction** (`feature_extraction/`)
   - `temporal_features.py`: Time-based patterns (burst detection, synchronization)
   - `text_features.py`: Comment similarity, spam detection, linguistic analysis
   - `network_features.py`: Graph analysis, community detection, cliques
   - `behavioral_features.py`: Account age, username patterns, automation detection

3. **Detection** (`detection/`)
   - `clustering.py`: HDBSCAN and DBSCAN ensemble clustering

4. **Visualization** (`visualization/`)
   - `network_viz.py`: Interactive Plotly graphs showing bot networks

5. **Storage** (`storage/`)
   - `database.py`: SQLite database for persistent storage

## Detection Methods

### 1. Temporal Burst Detection
Identifies accounts posting multiple comments in rapid succession:
```python
burst_score = rapid_posts / total_intervals
```
Accounts with burst_score > 0.5 are flagged as suspicious.

### 2. Synchronized Posting
Detects groups of accounts posting within the same time window (default: 5 minutes):
- Groups of 3+ accounts posting together
- Overlapping time windows are merged
- High synchronization indicates coordination

### 3. Text Similarity Analysis
Uses TF-IDF and Levenshtein distance to find:
- Template-based comments (similarity > 85%)
- Duplicate/near-duplicate content
- Low vocabulary diversity

### 4. Network Analysis
Builds multiple networks:
- **Co-occurrence network**: Accounts commenting on same videos
- **Reply network**: Directed graph of reply relationships
- **Temporal network**: Accounts posting close in time

Community detection using Louvain algorithm identifies tight-knit groups.

### 5. Behavioral Analysis
- **Account age**: New accounts (< 30 days) are suspicious
- **Username patterns**: Detects generated names (user12345, random strings)
- **Posting rate**: Abnormally high comment rates
- **Content targeting**: Accounts focusing on specific topics/channels

### 6. Ensemble Clustering
Combines multiple clustering runs:
- HDBSCAN with varying parameters
- DBSCAN with different epsilon values
- Consensus clustering for robust results

## Features Extracted

### Temporal Features (19 features)
- Comment count, active period, posting rate
- Mean/std/min/max inter-comment intervals
- Hour/day entropy (randomness of posting times)
- Burst score, regularity score

### Text Features (18 features)
- Average/std comment length
- Vocabulary size and richness
- Readability scores (Flesch-Kincaid)
- Sentiment scores (positive/negative/neutral)
- Special character ratios (!,?,CAPS)
- Template score, spam score, diversity score

### Network Features (15 features)
- Degree centrality, betweenness, eigenvector centrality
- PageRank score
- Clustering coefficient
- Community membership and size
- Clique membership
- Star pattern involvement

### Behavioral Features (12 features)
- Account age score
- Username pattern score
- Subscriber count, video count, view count
- Automation score (posting regularity)
- Content targeting score

## API Usage Examples

### Basic Usage
```bash
# Analyze specific videos
python main.py --urls "https://youtube.com/watch?v=VIDEO_ID" --max-comments 1000

# Analyze political content
python main.py --mode political --max-comments 5000

# Use labeled data for semi-supervised learning
python main.py --mode political --labeled-data data/labeled/bot_labels.csv
```

### Python API
```python
from main import YouTubeBotnetDetector

# Initialize detector
detector = YouTubeBotnetDetector()

# Collect data
comments_df = detector.collect_data(
    mode='urls',
    urls=['VIDEO_URL_1', 'VIDEO_URL_2'],
    max_comments=1000
)

# Extract features
features_df = detector.extract_all_features(comments_df)

# Detect bots
results_df = detector.detect_bots(features_df)

# Get bot accounts
bots = results_df[results_df['classification'] == 'likely_bot']
print(f"Found {len(bots)} bot accounts")
```

## Customization Guide

### Adjusting Detection Sensitivity
Edit `config/config.py`:

```python
# Make detection more sensitive (catch more bots)
BOT_PROBABILITY_THRESHOLD = 0.6  # Lower from 0.7
SUSPICIOUS_PROBABILITY_THRESHOLD = 0.4  # Lower from 0.5

# Adjust clustering parameters
MIN_CLUSTER_SIZE = 2  # Lower to detect smaller groups
MIN_EDGE_WEIGHT = 1  # Lower to create more network connections
```

### Adding Custom Features
1. Create new method in appropriate feature module
2. Add feature to compilation method
3. Update feature weights in config

Example:
```python
# In feature_extraction/custom_features.py
def extract_emoji_spam(comments_df):
    """Detect excessive emoji usage"""
    emoji_scores = {}
    for author_id, group in comments_df.groupby('author_id'):
        emoji_ratio = group['text'].str.count('[\U0001F300-\U0001F6FF]').sum() / len(group)
        emoji_scores[author_id] = min(emoji_ratio / 10, 1.0)  # Normalize
    return emoji_scores
```

### Custom Detection Rules
Add domain-specific rules in `detection/clustering.py`:

```python
def identify_bot_clusters(self, features_df, labels, threshold_features=None):
    # Add custom rules
    if 'domain_specific_feature' in features_df.columns:
        suspicious = features_df['domain_specific_feature'] > threshold
        # Adjust bot scores accordingly
```

## Understanding Results

### Output Files
1. **bot_detection_results_TIMESTAMP.csv**: Complete detection results
   - author_id: YouTube channel ID
   - cluster_id: Assigned cluster (-1 = noise)
   - final_bot_probability: 0-1 score (higher = more bot-like)
   - classification: likely_bot/suspicious/likely_human

2. **detection_summary_TIMESTAMP.json**: Statistical summary
   - Total accounts and classification breakdown
   - Top bot accounts with probabilities
   - Cluster statistics

3. **Visualizations**:
   - bot_network.html: Interactive network graph
   - cluster_analysis.html: Cluster comparison charts
   - temporal_patterns.html: Time-based activity patterns

### Interpreting Bot Probability
- **0.0-0.5**: Likely human
- **0.5-0.7**: Suspicious, needs review
- **0.7-1.0**: Likely bot

### Network Graph Colors
- 🔴 Red nodes: High bot probability
- 🟡 Yellow nodes: Suspicious accounts
- 🟢 Green nodes: Likely human
- Thicker edges: Stronger connections (more co-occurrences)

### Common Bot Patterns
1. **Burst Clusters**: Tight group posting within minutes
2. **Template Farms**: Multiple accounts using similar text
3. **Engagement Rings**: Accounts that always appear together
4. **Spam Networks**: Promoting links/products
5. **Astroturfing**: Coordinated political/commercial messaging

## Limitations & Ethical Considerations

### Limitations
- API quotas limit data collection (10,000 units/day free tier)
- Cannot detect sophisticated bots mimicking human behavior
- False positives possible for legitimate coordinated campaigns
- Temporal patterns may miss bots posting across long time periods

### Ethical Use
- This tool is for educational and research purposes only
- Do not use for harassment or doxxing
- Verify findings before taking action
- Respect privacy and YouTube's Terms of Service
- Consider false positive implications

### Improving Accuracy
1. Collect more data over longer time periods
2. Use multiple API keys for higher quotas
3. Incorporate labeled data for semi-supervised learning
4. Adjust thresholds based on domain knowledge
5. Combine with other signals (video metadata, user reports)

## Troubleshooting

### Common Issues

**API Key Errors**:
```
ValueError: No YouTube API keys found
```
Solution: Copy `.env.template` to `.env` and add your API key

**Import Errors**:
```
ModuleNotFoundError: No module named 'hdbscan'
```
Solution: Install requirements: `pip install -r requirements.txt`

**Memory Issues with Large Datasets**:
- Reduce max_comments parameter
- Process videos in batches
- Use database for intermediate storage

**No Bots Detected**:
- Lower detection thresholds in config
- Ensure sufficient data collected (need patterns to emerge)
- Check if comments are from diverse time periods

## Further Development Ideas

1. **Machine Learning Enhancement**:
   - Train supervised classifier on labeled data
   - Use deep learning for text analysis
   - Implement active learning for labeling

2. **Additional Features**:
   - Profile picture analysis
   - Video watching patterns
   - Cross-platform detection

3. **Real-time Monitoring**:
   - Webhook integration
   - Dashboard for continuous monitoring
   - Alert system for bot campaigns

4. **Scalability**:
   - Distributed processing
   - Cloud deployment
   - API endpoint creation

## References

- YouTube Data API v3 Documentation
- HDBSCAN: Hierarchical Density-Based Spatial Clustering
- Louvain Community Detection Algorithm
- Network Analysis with NetworkX
- Botnet Detection in Social Media (Research Papers)
