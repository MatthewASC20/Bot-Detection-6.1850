# YouTube Bot Detector - Chrome Extension

A Chrome extension that detects bot comments on YouTube using machine learning-based probability scores and community voting.

## Features

- **Real-time Bot Detection**: Automatically analyzes YouTube comments and displays bot probability scores
- **Visual Indicators**: Color-coded risk levels (Low/Medium/High) for easy identification
- **Community Voting**: Upvote/downvote comments as bot or human to improve detection
- **Local & Backend Analysis**: Works offline with heuristics or connects to your ML backend
- **Privacy-Focused**: All voting data stored locally in your browser
- **Seamless Integration**: Non-intrusive UI that fits naturally into YouTube's design

## Screenshots

The extension adds bot probability indicators directly to YouTube comments:
- 🟢 **Low Risk** (0-40%): Green indicator
- 🟠 **Medium Risk** (40-70%): Orange indicator
- 🔴 **High Risk** (70-100%): Red indicator

Each comment shows:
- Bot probability percentage
- Vote buttons (Bot/Human)
- Your previous votes are remembered

## Installation

### 1. Install the Extension

1. Download or clone this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" (toggle in top-right corner)
4. Click "Load unpacked"
5. Select the `youtube-bot-detector-extension` folder

### 2. Set Up the Backend (Optional but Recommended)

The extension works with local heuristics by default, but for better accuracy, you can run the Flask backend:

```bash
cd backend
pip install -r requirements.txt
python api.py
```

The API will run on `http://localhost:5001`

### 3. Configure the Extension

1. Click the extension icon in Chrome toolbar
2. Set the API endpoint (default: `http://localhost:5001/api`)
3. Test the connection
4. Save settings

## Usage

### Browsing YouTube

1. Navigate to any YouTube video
2. Scroll to the comments section
3. Bot indicators will automatically appear on comments
4. Click voting buttons to mark comments as bot or human

### Voting System

- **👍 Bot**: Mark this comment as likely from a bot
- **👤 Human**: Mark this comment as from a real person
- Click again to remove your vote
- Your votes are saved and synced (optional) with the backend

### Settings

Access settings by clicking the extension icon:

- **Enable Detection**: Turn bot detection on/off
- **Enable Voting**: Show/hide voting buttons
- **API Endpoint**: Configure backend URL
- **Clear Data**: Remove all stored votes

## Architecture

### Extension Components

```
youtube-bot-detector-extension/
├── manifest.json          # Extension configuration
├── content.js            # Main detection logic
├── background.js         # Service worker for API calls
├── popup.html/js         # Settings interface
├── styles.css            # UI styling
├── icons/                # Extension icons
└── backend/              # Flask API (optional)
    ├── api.py           # REST API server
    └── requirements.txt  # Python dependencies
```

### Detection Methods

#### Local Heuristics (No Backend Required)
- Username pattern analysis (numbers, length, caps)
- Spam keyword detection
- Link detection
- Emoji frequency
- Comment length and repetition
- Generic phrase detection

#### Backend ML Detection (With API)
- All local features plus:
- Advanced NLP analysis
- Temporal pattern detection
- Network analysis
- Behavioral profiling
- Community vote aggregation

## Integrating with Your Bot Detection System

To use your existing bot detection system, modify `backend/api.py`:

```python
# Import your detection modules
from detection.clustering import BotDetector
from feature_extraction.text_features import extract_text_features

# In the analyze_comment() function:
def analyze_comment():
    data = request.json
    
    # Use your existing feature extraction
    features = extract_text_features(data)
    
    # Use your detector
    detector = BotDetector()
    probability = detector.predict_single(features)
    
    return jsonify({'bot_probability': probability})
```

## API Endpoints

### `POST /api/analyze`
Analyze a comment for bot probability

**Request:**
```json
{
  "author": "username",
  "content": "comment text",
  "timestamp": "2 hours ago",
  "likes": "5"
}
```

**Response:**
```json
{
  "bot_probability": 0.85,
  "features": {...},
  "community_votes": {
    "bot_votes": 3,
    "human_votes": 1
  }
}
```

### `POST /api/vote`
Submit a user vote

**Request:**
```json
{
  "commentId": "unique-id",
  "vote": 1,  // 1 = bot, -1 = human, 0 = neutral
  "commentData": {...}
}
```

### `GET /api/stats`
Get overall statistics

**Response:**
```json
{
  "total_votes": 150,
  "bot_votes": 90,
  "human_votes": 60
}
```

### `GET /api/health`
Health check endpoint

## Privacy & Data

- All voting data is stored locally in Chrome's storage
- Backend communication is optional
- No personal information is collected
- Votes are anonymous
- Data can be cleared anytime from settings

## Development

### Local Development

1. Make changes to extension files
2. Go to `chrome://extensions/`
3. Click refresh icon on the extension card
4. Reload YouTube page to see changes

### Testing

Test the extension on various YouTube videos:
- Popular videos with many comments
- Videos with known bot activity
- Live stream chats
- Different comment sorting (Top/Newest)

### Debugging

- Open Chrome DevTools (F12) on YouTube
- Check Console for extension logs
- Inspect Network tab for API calls
- Use `chrome://extensions/` to view background errors

## Troubleshooting

### Extension Not Detecting Comments

- Ensure the extension is enabled
- Check that detection is enabled in settings
- Refresh the YouTube page
- Check browser console for errors

### Backend Connection Failed

- Verify the API is running: `python api.py`
- Check the API endpoint in settings
- Test connection using the "Test Connection" button
- Ensure no firewall is blocking port 5000
- Check for CORS errors in console

### Votes Not Saving

- Check Chrome storage is not full
- Verify voting is enabled in settings
- Look for errors in browser console
- Try clearing extension data and re-voting

## Future Enhancements

- [ ] Machine learning model integration
- [ ] Batch comment analysis
- [ ] Export voting data for training
- [ ] Community consensus scoring
- [ ] Report suspected bot networks
- [ ] Integration with your existing HDBSCAN clustering
- [ ] Temporal burst detection visualization
- [ ] Network graph overlay on YouTube

## Contributing

Contributions are welcome! Areas for improvement:
- Better ML model integration
- UI/UX enhancements
- Additional detection heuristics
- Performance optimizations
- Testing and documentation

## License

This project is for educational purposes. Use responsibly and in accordance with YouTube's Terms of Service.

## Credits

Built to integrate with comprehensive YouTube bot detection system featuring:
- Ensemble clustering (HDBSCAN/DBSCAN)
- Network analysis with community detection
- Temporal pattern recognition
- Multi-dimensional behavioral profiling

---

**Note**: This extension is a proof-of-concept for educational purposes. Always verify bot detection results and use good judgment when moderating content.
