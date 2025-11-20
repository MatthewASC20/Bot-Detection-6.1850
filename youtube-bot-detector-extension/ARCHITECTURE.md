# YouTube Bot Detector - System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         YouTube Website                          │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                    Comments Section                      │    │
│  │                                                          │    │
│  │  ┌──────────────────────────────────────────┐          │    │
│  │  │ 🤖 Bot Risk: High 85%  [👍Bot][👤Human] │ ◄────┐   │    │
│  │  └──────────────────────────────────────────┘      │   │    │
│  │                                                     │   │    │
│  │  ┌──────────────────────────────────────────┐      │   │    │
│  │  │ 🤖 Bot Risk: Low 12%  [👍Bot][👤Human]  │ ◄────┤   │    │
│  │  └──────────────────────────────────────────┘      │   │    │
│  └─────────────────────────────────────────────────────┘   │    │
└─────────────────────────────────────────────────────────────────┘
                                                             │
                              Injected by Content Script ────┘
                                                             │
                                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               Chrome Extension (Client-Side)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  content.js  │  │ background.js│  │  popup.html  │         │
│  │              │  │              │  │              │         │
│  │ • Observe    │  │ • API calls  │  │ • Settings   │         │
│  │ • Extract    │  │ • Storage    │  │ • Stats      │         │
│  │ • Analyze    │  │ • Messages   │  │ • Config     │         │
│  │ • Render     │  │ • Cleanup    │  │              │         │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘         │
│         │                  │                                     │
│         └──────────────────┼─────────────────────┐              │
│                            │                     │              │
└────────────────────────────┼─────────────────────┼──────────────┘
                             │                     │
                             │ HTTP/REST           │ Chrome Storage
                             │                     │
                             ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend API (Optional)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────┐            │
│  │              Flask REST API                     │            │
│  │                                                 │            │
│  │  GET  /api/health      - Health check          │            │
│  │  POST /api/analyze     - Analyze comment       │            │
│  │  POST /api/vote        - Submit vote           │            │
│  │  GET  /api/stats       - Get statistics        │            │
│  └────────────────┬───────────────────────────────┘            │
│                   │                                             │
│                   ▼                                             │
│  ┌────────────────────────────────────────────────┐            │
│  │         Your Bot Detection System               │            │
│  │                                                 │            │
│  │  • Feature Extraction (text, temporal, etc.)   │            │
│  │  • HDBSCAN/DBSCAN Clustering                   │            │
│  │  • Network Analysis (Louvain)                  │            │
│  │  • Temporal Burst Detection                    │            │
│  │  • Behavioral Profiling                        │            │
│  └────────────────┬───────────────────────────────┘            │
│                   │                                             │
│                   ▼                                             │
│  ┌────────────────────────────────────────────────┐            │
│  │           SQLite Database                       │            │
│  │                                                 │            │
│  │  • votes (comment_id, vote, timestamp)         │            │
│  │  • statistics (aggregated data)                │            │
│  └────────────────────────────────────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

### 1. Comment Detection & Analysis

```
User visits YouTube video
          ↓
Content script observes DOM
          ↓
New comments detected
          ↓
For each comment:
    │
    ├─→ Extract comment data
    │   (author, content, timestamp, etc.)
    │
    ├─→ Calculate local heuristic score
    │   (username patterns, spam keywords, etc.)
    │
    ├─→ [Optional] Send to backend API
    │   POST /api/analyze
    │        ↓
    │   Feature extraction
    │        ↓
    │   ML model prediction
    │        ↓
    │   Return probability
    │
    └─→ Render UI indicator
        (color-coded badge + vote buttons)
```

### 2. Voting Flow

```
User clicks vote button
          ↓
Toggle vote state
(1 = bot, -1 = human, 0 = neutral)
          ↓
Update local UI
(highlight active button)
          ↓
Save to Chrome storage
          ↓
[Optional] Sync to backend
POST /api/vote
    ↓
Store in database
    ↓
Update community stats
```

### 3. Settings Management

```
User opens popup
          ↓
Load current settings
from Chrome storage
          ↓
Display stats & config
          ↓
User changes settings
          ↓
Save to Chrome storage
          ↓
[Optional] Test connection
GET /api/health
          ↓
Display result to user
```

## Data Structures

### Comment Data Object
```javascript
{
  author: "string",          // Username
  authorLink: "url",         // Channel URL
  content: "string",         // Comment text
  timestamp: "string",       // Relative time
  likes: "string",           // Like count
  isPinned: boolean,         // Is pinned
  isHeartedByCreator: boolean // Creator heart
}
```

### Vote Data Object
```javascript
{
  commentId: "string",       // Unique ID
  vote: number,              // 1, -1, or 0
  commentData: {...},        // Full comment data
  timestamp: number          // Unix timestamp
}
```

### Analysis Response
```javascript
{
  bot_probability: float,    // 0.0 to 1.0
  features: {...},           // Extracted features
  community_votes: {         // Vote aggregation
    bot_votes: number,
    human_votes: number
  },
  confidence: float,         // Model confidence (optional)
  method: "string"           // "heuristic" or "ml_model"
}
```

## Storage Architecture

### Chrome Local Storage
```
storage.local {
  votingData: {
    "comment-id-1": {
      vote: 1,
      timestamp: 1234567890,
      commentData: {...}
    },
    "comment-id-2": {...}
  },
  apiEndpoint: "http://localhost:5000/api",
  detectionEnabled: true,
  votingEnabled: true,
  thresholds: {
    low: 0.3,
    medium: 0.5,
    high: 0.7
  }
}
```

### Backend Database (SQLite)
```sql
CREATE TABLE votes (
    id INTEGER PRIMARY KEY,
    comment_id TEXT NOT NULL,
    vote INTEGER NOT NULL,      -- 1 or -1
    author TEXT,
    content TEXT,
    timestamp INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_comment_id ON votes(comment_id);
```

## Security Architecture

### Content Security Policy
- Scripts: Only from extension
- Styles: Inline allowed (scoped)
- Images: Extension resources only
- Network: API endpoint only

### Data Protection
- Local storage: Chrome's secure storage
- API calls: HTTPS in production
- Input validation: All endpoints
- Rate limiting: Prevent abuse
- CORS: Configured for extension

### Privacy Measures
- No external tracking
- Local-first design
- Optional backend sync
- User-controlled data
- Clear data mechanism

## Extension Lifecycle

```
Installation
    ↓
Initialize storage with defaults
    ↓
Load icon assets
    ↓
Register content scripts
    ↓
Start background service worker
    ↓
─────────────────────────────
    ↓
User navigates to YouTube
    ↓
Content script injected
    ↓
Observe for comments
    ↓
Process & analyze
    ↓
Render indicators
    ↓
Listen for interactions
    ↓
─────────────────────────────
    ↓
Daily cleanup alarm
(remove votes > 30 days old)
```

## Performance Optimization

### Client-Side
- Lazy loading: Only process visible comments
- Debouncing: Limit DOM observations
- Caching: Remember processed comments
- Batch updates: Group DOM manipulations

### Backend
- Connection pooling: Reuse database connections
- Query optimization: Indexed lookups
- Response caching: Cache common requests
- Async processing: Non-blocking I/O

## Error Handling Strategy

```
Operation Attempt
    ↓
Try local operation
    ↓
    ├─→ Success: Render result
    │
    └─→ Failure: Try backend
            ↓
            ├─→ Success: Render result
            │
            └─→ Failure: Fallback to safe default
                    ↓
                Log error (console)
                    ↓
                Show user-friendly message
```

## Scalability Considerations

### Current Scale
- Single user
- Local processing
- < 1000 comments per page
- SQLite database

### Future Scale
- Multi-user backend
- Distributed processing
- Cloud database (PostgreSQL)
- Caching layer (Redis)
- Load balancing
- Microservices architecture

## Integration Points

### With Your Bot Detection System

```
Extension → API → Your System

POST /api/analyze
    ↓
feature_extraction/
    ├─ text_features.py
    ├─ temporal_features.py
    ├─ network_features.py
    └─ behavioral_features.py
    ↓
detection/
    └─ clustering.py
        (HDBSCAN/DBSCAN)
    ↓
Return probability
```

## Technology Stack Summary

**Frontend (Extension)**
- JavaScript ES6+
- Chrome Extension API v3
- CSS3 (with animations)
- HTML5

**Backend (API)**
- Python 3.7+
- Flask web framework
- Flask-CORS
- SQLite database

**Your Bot Detection**
- scikit-learn
- HDBSCAN/DBSCAN
- NetworkX
- Pandas/NumPy
- TF-IDF/NLP libraries

---

This architecture ensures:
✅ Scalability
✅ Maintainability  
✅ Security
✅ Performance
✅ Extensibility
✅ Privacy
