# YouTube Bot Detector - Quick Start Guide

## 🚀 Installation (3 minutes)

### Step 1: Start the Backend
```bash
cd youtube-bot-detector-extension/backend
pip install -r requirements.txt
python api.py
```

You should see:
```
Starting YouTube Bot Detector API...
API will be available at: http://localhost:5000
```

### Step 2: Load Extension in Chrome

1. Open Chrome and navigate to: `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked"
4. Select the `youtube-bot-detector-extension` folder
5. You should see the extension icon appear in your toolbar

### Step 3: Test It Out

1. Click the extension icon 
2. Click "Test Connection" - should see ✓ Connection successful
3. Navigate to any YouTube video
4. Scroll to comments
5. You'll see bot probability indicators appear!

## 📊 What You'll See

Each YouTube comment will show:

```
┌─────────────────────────────────────┐
│ 🤖 Bot Risk: High 85%               │
│ [👍 Bot] [👤 Human]                 │
└─────────────────────────────────────┘
```

### Risk Levels

🟢 **Low (0-40%)** - Likely human
🟠 **Medium (40-70%)** - Uncertain  
🔴 **High (70-100%)** - Likely bot

## 🎯 How to Use

### Voting
- Click **👍 Bot** if you think it's a bot
- Click **👤 Human** if you think it's human
- Click again to remove your vote

### View Your Stats
Click the extension icon to see:
- Total votes cast
- Bot vs Human breakdown
- Settings and configuration

## 🔧 Integration with Your Bot Detection System

To use your existing detection system instead of simple heuristics:

1. Open `backend/integration_example.py`
2. Uncomment the imports for your detection modules
3. Adjust paths to match your project structure
4. Run: `python integration_example.py` instead of `api.py`

Example integration:
```python
from detection.clustering import BotDetector
from feature_extraction.text_features import extract_text_features

detector = BotDetector()
features = extract_text_features(comment_data)
probability = detector.predict_single(features)
```

## 🐛 Troubleshooting

**Extension not showing on YouTube?**
- Refresh the YouTube page
- Check extension is enabled at chrome://extensions/
- Check browser console (F12) for errors

**Bot indicators not appearing?**
- Ensure "Enable Detection" is ON in extension settings
- Wait a few seconds for comments to load
- Try scrolling to load more comments

**Backend connection failed?**
- Check API is running: http://localhost:5000/api/health
- Verify endpoint in extension settings
- Check firewall isn't blocking port 5000

## 💡 Tips

1. **Best Results**: Use with your ML model for accurate detection
2. **Community Data**: Your votes help improve the system over time
3. **Privacy**: All data stored locally unless you enable backend sync
4. **Performance**: Works on videos with 1000+ comments

## 📈 Advanced Features

### Batch Analysis
Analyze multiple comments at once for network detection:
```bash
curl -X POST http://localhost:5000/api/analyze_batch \
  -H "Content-Type: application/json" \
  -d '{"comments": [...]}'
```

### Network Analysis
Detect coordinated bot networks:
```bash
curl -X POST http://localhost:5000/api/analyze_network \
  -H "Content-Type: application/json" \
  -d '{"comment": {...}, "video_id": "..."}'
```

## 📚 Learn More

- **Full Documentation**: See README.md
- **API Reference**: Check backend/api.py for all endpoints
- **Integration Guide**: See backend/integration_example.py
- **Your Bot Detection System**: Leverage your existing HDBSCAN clustering, network analysis, and temporal detection

---

**Questions?** Check the README.md or browser console for debugging info.

**Privacy Note**: This extension respects your privacy. Voting data is stored locally. Backend communication is optional and can be disabled.
