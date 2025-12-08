# YouTube Botnet Detection System – Technical Documentation

Comprehensive notes for the full pipeline, feature catalog, and supporting services in this repository.

## 1) Architecture & Flow
```
YouTube API → data_collection → features/ → detection/ → visualization/ & storage/
                                  ↑
                           discovery/ + api/ (extension/batch)
```
- **data_collection/** – API wrapper with key rotation, caching, channel/video enrichment, and collection helpers (`urls`, `political`, `search` modes).
- **features/** – temporal, text/NLP, network, behavioral, and semantic extractors used by the clustering pipeline; also reused by the extension API.
- **detection/** – two detectors:
  - `BotBusterDetector` (default): semantic + temporal coordination, cross-video similarity, comment-level probabilities, coordination graph clustering.
  - `ClusteringDetector`: ensemble HDBSCAN/DBSCAN over the full feature set with individual + cluster scores.
- **visualization/** – Plotly/Matplotlib HTML reports (network, clusters, temporal, cross-video, summaries).
- **storage/** – SQLite schema for comments/videos/channels/detection runs; additional tables for extension votes & discovery artifacts.
- **discovery/** – expansion crawler + botnet identifier for following suspicious accounts and grouping discovered clusters.
- **api/** – Flask API powering the Chrome extension and batch `/api/analyze_video`, reusing the same feature pipeline.

## 2) Detection methods
### BotBuster (default path in `main.py --method botbuster`)
1. Build temporal synchronization matrix (same-video, within `Config.TEMPORAL_WINDOW_SECONDS`).
2. Compute semantic similarity (OpenAI embeddings if available; TF-IDF fallback).
3. Boost cross-video semantic similarity (strong coordination signal).
4. Combine into a coordination matrix; zero out self/self-author edges.
5. Build an author coordination graph and cluster with DBSCAN to find botnets.
6. Score comments and accounts, weighting cross-video similarity, coordination, cluster size, and average comment probability.
7. Produce comment-level CSV (probabilities), account-level CSV/JSON summaries, and visualizations (network, heatmap, cross-video, summary).

### Original clustering (`--method original`)
1. Extract the full feature set (temporal, text, network, behavioral, semantic).
2. Standardize + PCA (95% variance) when needed.
3. Ensemble of HDBSCAN/DBSCAN runs; consensus clustering + confidence.
4. Cluster-level bot scores blended with weighted individual feature scores (see `Config.FEATURE_WEIGHTS`).

### Comparison
`run_comparison.py` runs both detectors on the same dataset and saves side-by-side reports.

## 3) Feature catalog (columns produced by `extract_all_features`)

**Temporal**
- `comment_count`, `first_comment`, `last_comment`, `active_period_hours`, `comments_per_hour`
- `most_active_hour`, `hour_entropy`, `most_active_day`, `day_entropy`
- `mean_interval_seconds`, `std_interval_seconds`, `min_interval_seconds`, `max_interval_seconds`
- `burst_score`, `regularity_score`

**Text / NLP**
- `avg_comment_length`, `std_comment_length`, `total_comments`
- `vocabulary_size`, `vocabulary_richness`
- `flesch_reading_ease`, `flesch_kincaid_grade`
- Sentiment: `avg_sentiment_compound`, `std_sentiment_compound`, `avg_sentiment_positive`, `avg_sentiment_negative`, `avg_sentiment_neutral`
- Character/formatting: `exclamation_ratio`, `question_ratio`, `caps_ratio`, `emoji_count`, `url_count`
- Repetition: `repeated_words_ratio`, `repeated_phrases_count`
- Pattern flags: `template_score`, `spam_score`, `diversity_score`

**Network (co-occurrence, replies, temporal proximity)**
- Co-occurrence metrics prefixed `co_`: `co_degree`, `co_weighted_degree`, `co_degree_centrality`, `co_betweenness_centrality`, `co_eigenvector_centrality`, `co_pagerank`, `co_clustering_coefficient`, `co_neighbor_count`
- `community_id`, `community_size`
- `replies_sent`, `replies_received`, `temporal_degree`
- Clique/star flags: `in_clique`, `max_clique_size`, `is_star_center`, `in_star_pattern`

**Behavioral**
- Channel stats: `subscriber_count`, `video_count`, `total_views`, `views_per_video`, `subscriber_view_ratio`
- Activity: `total_comments`, `unique_videos_commented`, `comments_per_video`
- Engagement: `total_likes_received`, `avg_likes_per_comment`, `zero_like_ratio`, `reply_ratio`
- Scores: `account_age_score`, `username_pattern_score`, `automation_score`, `targeting_score`

**Semantic (embeddings + local fallbacks)**
- `avg_semantic_similarity_to_others`, `max_semantic_similarity_to_others`, `high_similarity_count`
- `avg_discussion_similarity`, `semantic_diversity`
- Topics/intent: `dominant_topic`, `topic_concentration`, `intent_bot_score`, `intent_human_score`
- Coordinated language: `talking_point_matches`, `uses_coordinated_language`

Additional detectors (not direct columns) flag synchronized posting windows, duplicate comments, and coordinated talking points for UI/alerts.

## 4) Running the pipeline
CLI (`main.py`):
```bash
# BotBuster (semantic + temporal coordination, default)
python main.py --mode urls --urls "VIDEO1,VIDEO2" --max-comments 800 --method botbuster

# Feature-based clustering
python main.py --mode political --max-comments 500 --method original

# Run both for comparison
python main.py --mode search --method both --max-comments 400
```
Important flags: `--no-cache`, `--clear-db`, `--labeled-data <csv>`, and `--mode {urls,political,search}`.

API / Extension:
- `python api/integrated_api.py` starts Flask on `:5001` with `/api/analyze_video` (batch analysis), `/api/analyze` (legacy single comment), `/api/vote`, `/api/stats`, `/api/author/<id>`, `/api/health`.
- Extension docs live in `youtube-bot-detector-extension/documentation/`.

Discovery:
- `discovery/botnet_discovery.py` orchestrates seed collection, expansion (`ExpansionCrawler`), reclustering, and botnet grouping (`BotnetIdentifier`). Exposed via the API dashboard endpoints for visualization.

## 5) Outputs & storage
- SQLite at `data/botnet_detection.db`: comments, videos, channels, detection_results, discovery runs, extension votes/stats.
- Reports (`outputs/reports/`): `bot_detection_results_<ts>.csv`, `detection_summary_<ts>.json`, comment-level probabilities for BotBuster, comparison reports.
- Visuals (`outputs/graphs/`): `bot_network.html`, `cluster_analysis.html`, `temporal_patterns.html`, plus BotBuster-specific `coordination_heatmap.html`, `cross_video_analysis.html`, `detection_summary.html` when applicable.

## 6) Configuration notes
- `config/config.py` controls API keys, clustering parameters, thresholds, directory creation, and feature weights. Missing `YOUTUBE_API_KEY` raises immediately.
- OpenAI is optional; when not configured, `SemanticFeatures` uses TF-IDF/topics/intents to keep the pipeline running offline.
- Adjust `BOT_PROBABILITY_THRESHOLD`, `SUSPICIOUS_PROBABILITY_THRESHOLD`, `MIN_CLUSTER_SIZE`, `MIN_EDGE_WEIGHT`, and `TEMPORAL_WINDOW_SECONDS` to tune sensitivity.

## 7) Limitations & troubleshooting
- **API quota**: rely on caching and multiple API keys for large runs.
- **Dependency issues**: install via `requirements.txt`; download NLTK corpora (`punkt`, `stopwords`, `vader_lexicon`).
- **No bots detected**: lower thresholds, increase `max-comments`, or add more diverse videos to expose coordination.
- **Memory/time**: reduce per-video comment caps or run in batches; BotBuster embeddings on very large corpora can be expensive.

Use responsibly for research and moderation support. Always verify findings and respect YouTube’s Terms of Service.
