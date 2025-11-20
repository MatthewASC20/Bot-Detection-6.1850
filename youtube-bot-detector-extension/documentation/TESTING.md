# Testing Checklist for YouTube Bot Detector

## Pre-Launch Testing

### ✅ Installation Testing

- [ ] Extension loads without errors in Chrome
- [ ] All icons display correctly (16x16, 32x32, 48x48, 128x128)
- [ ] Manifest.json is valid
- [ ] No console errors on installation
- [ ] Extension appears in Chrome toolbar
- [ ] Extension popup opens correctly

### ✅ Backend Testing

- [ ] Flask server starts without errors
- [ ] `/api/health` endpoint responds
- [ ] `/api/analyze` accepts POST requests
- [ ] `/api/vote` records votes correctly
- [ ] `/api/stats` returns accurate statistics
- [ ] CORS headers are set correctly
- [ ] Database initializes properly
- [ ] Error handling works (try invalid JSON)

### ✅ Core Functionality

#### Comment Detection
- [ ] Indicators appear on YouTube comments
- [ ] Color coding works (green/orange/red)
- [ ] Percentages display correctly
- [ ] Works on comment refresh/scroll
- [ ] Works on different video pages
- [ ] No duplicate indicators on same comment

#### Probability Calculation
- [ ] Obvious bots score high (>70%)
- [ ] Normal comments score low (<40%)
- [ ] Edge cases score medium (40-70%)
- [ ] Backend probabilities match frontend display
- [ ] Local heuristics work when offline

#### Voting System
- [ ] Vote buttons appear and are clickable
- [ ] "Bot" vote highlights correctly
- [ ] "Human" vote highlights correctly
- [ ] Clicking same button toggles vote off
- [ ] Votes persist across page refreshes
- [ ] Vote feedback message displays
- [ ] Votes sync to backend (if connected)

### ✅ UI/UX Testing

#### Visual Design
- [ ] Matches YouTube's design language
- [ ] Works in light mode
- [ ] Works in dark mode (YouTube dark theme)
- [ ] Responsive on different screen sizes
- [ ] No UI overflow or clipping
- [ ] Smooth animations and transitions
- [ ] Icons are crisp and clear

#### Popup Interface
- [ ] Statistics display correctly
- [ ] Settings toggles work
- [ ] API endpoint field accepts input
- [ ] Test connection button works
- [ ] Save button persists settings
- [ ] Clear data button clears votes
- [ ] Status messages appear and disappear

### ✅ Edge Cases

#### YouTube Variations
- [ ] Works on video watch pages
- [ ] Works on live streams
- [ ] Works on premiere videos
- [ ] Works with pinned comments
- [ ] Works with creator-hearted comments
- [ ] Works with nested replies
- [ ] Handles very long comments
- [ ] Handles very short comments
- [ ] Handles comments with emojis
- [ ] Handles comments with links

#### Browser Scenarios
- [ ] Works after browser restart
- [ ] Works in incognito mode
- [ ] Multiple YouTube tabs
- [ ] Fast page navigation
- [ ] Back/forward browser buttons
- [ ] Hard refresh (Ctrl+F5)

#### Network Scenarios
- [ ] Works offline (local heuristics)
- [ ] Graceful backend connection failure
- [ ] API timeout handling
- [ ] Slow network conditions
- [ ] Backend restart recovery

### ✅ Performance Testing

- [ ] No significant page load delay
- [ ] Handles 100+ comments smoothly
- [ ] Handles 1000+ comments acceptably
- [ ] No memory leaks (check Chrome Task Manager)
- [ ] CPU usage reasonable
- [ ] Network requests optimized
- [ ] No lag when scrolling
- [ ] Fast vote response time

### ✅ Data & Privacy

- [ ] Votes stored in Chrome local storage
- [ ] No sensitive data logged
- [ ] Data cleared when requested
- [ ] Old votes cleaned up (30-day rule)
- [ ] No tracking cookies
- [ ] API calls use HTTPS (if deployed)

### ✅ Error Handling

- [ ] Graceful handling of missing elements
- [ ] API errors don't break UI
- [ ] Invalid JSON responses handled
- [ ] Network timeouts handled
- [ ] Malformed comments handled
- [ ] Console errors are informative
- [ ] User-facing error messages clear

## Test Cases for Different Comment Types

### 🤖 Known Bot Comments (Should Score HIGH)

- [ ] "Click here for free money 💰💰💰"
- [ ] "Congratulations! Visit: sketchy-site.com"
- [ ] "Sub to my channel!!!! 🔥🔥🔥🔥🔥"
- [ ] Username: "User847362947"
- [ ] Repetitive: "aaaaaawesome video!!!!"
- [ ] Generic: "first"
- [ ] Generic: "nice video"

### 👤 Known Human Comments (Should Score LOW)

- [ ] Thoughtful discussion about video content
- [ ] Specific questions about timestamps
- [ ] Personal stories related to topic
- [ ] Constructive criticism
- [ ] Creator responses
- [ ] Detailed technical explanations

### ❓ Ambiguous Comments (Should Score MEDIUM)

- [ ] "Great video!"
- [ ] "Awesome content 👍"
- [ ] "Thanks for sharing"
- [ ] Short but relevant comments
- [ ] Comments with 1-2 emojis

## Integration Testing with Your Bot Detection System

### ✅ Feature Extraction
- [ ] Text features extracted correctly
- [ ] Temporal features calculated
- [ ] Network features computed
- [ ] Behavioral features identified
- [ ] Feature vector format matches model

### ✅ ML Model Integration
- [ ] Model loads without errors
- [ ] Predictions return valid probabilities (0-1)
- [ ] Batch predictions work
- [ ] Model inference time acceptable
- [ ] Confidence scores accurate

### ✅ Database Integration
- [ ] Comments stored correctly
- [ ] Votes linked to comments
- [ ] Queries return expected results
- [ ] No duplicate entries
- [ ] Database migrations work

## Browser Compatibility

- [ ] Chrome (latest)
- [ ] Chrome (one version back)
- [ ] Edge (Chromium-based)
- [ ] Brave
- [ ] Opera

## Security Testing

- [ ] No XSS vulnerabilities
- [ ] No injection attacks possible
- [ ] API endpoints validate input
- [ ] Rate limiting works (if implemented)
- [ ] No sensitive data in console logs
- [ ] Content Security Policy respected

## Accessibility Testing

- [ ] Keyboard navigation works
- [ ] Screen reader compatible
- [ ] Color contrast sufficient
- [ ] Focus indicators visible
- [ ] Alt text for images
- [ ] ARIA labels where needed

## Documentation Testing

- [ ] README.md accurate and complete
- [ ] Installation steps work as written
- [ ] API documentation matches implementation
- [ ] Code comments are helpful
- [ ] Examples run without errors
- [ ] Links work (no 404s)

## Final Pre-Release Checklist

- [ ] All critical bugs fixed
- [ ] Performance acceptable
- [ ] Documentation complete
- [ ] Examples working
- [ ] Code cleaned up (no debug logs)
- [ ] Version number updated
- [ ] License file included
- [ ] Screenshots/demos ready
- [ ] Privacy policy clear
- [ ] Terms of service (if needed)

## Post-Launch Monitoring

### Week 1
- [ ] Monitor error logs
- [ ] Check API usage patterns
- [ ] Gather user feedback
- [ ] Watch for edge cases
- [ ] Track performance metrics

### Week 2-4
- [ ] Analyze voting patterns
- [ ] Identify false positives/negatives
- [ ] Collect feature requests
- [ ] Plan improvements
- [ ] Optimize based on real usage

## Regression Testing (After Updates)

After any code changes, re-test:
- [ ] Core detection functionality
- [ ] Voting system
- [ ] Settings persistence
- [ ] API communication
- [ ] UI rendering
- [ ] Performance benchmarks

## Load Testing Scenarios

### Light Load
- [ ] 1 user, 50 comments - response time < 100ms
- [ ] Vote submission < 200ms

### Medium Load
- [ ] 10 users, 500 comments - no degradation
- [ ] Backend handles concurrent requests

### Heavy Load
- [ ] 100 users, 5000 comments - graceful degradation
- [ ] Database queries optimized
- [ ] API rate limiting effective

## User Acceptance Testing

Get feedback on:
- [ ] Ease of installation
- [ ] Clarity of UI
- [ ] Accuracy of detection
- [ ] Usefulness of voting
- [ ] Overall experience
- [ ] Feature requests
- [ ] Bug reports

---

## Bug Report Template

```markdown
**Description**: 
**Steps to Reproduce**:
1. 
2. 
3. 

**Expected Behavior**: 
**Actual Behavior**: 
**Browser**: 
**Extension Version**: 
**Screenshots**: 
**Console Errors**: 
```

---

## Testing Notes

- Test regularly during development
- Automate where possible
- Keep test data diverse
- Document unexpected behavior
- Update tests when adding features
- Don't skip edge cases
- Real-world testing is essential

**Remember**: Testing is not just about finding bugs, it's about ensuring the extension actually helps users identify bots effectively!
