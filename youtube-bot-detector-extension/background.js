// Background service worker for the YouTube Bot Detector extension

// Listen for extension installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('YouTube Bot Detector installed!');
    // Set default settings
    chrome.storage.local.set({
      apiEndpoint: 'http://localhost:5001/api',
      detectionEnabled: true,
      votingEnabled: true,
      thresholds: {
        low: 0.3,
        medium: 0.5,
        high: 0.7
      }
    });
  } else if (details.reason === 'update') {
    console.log('YouTube Bot Detector updated!');
  }
});

// Handle messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'analyzeComment') {
    analyzeComment(request.data)
      .then(result => sendResponse({ success: true, data: result }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true; // Indicates async response
  }
  
  if (request.action === 'submitVote') {
    submitVote(request.data)
      .then(result => sendResponse({ success: true, data: result }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
  
  if (request.action === 'getStats') {
    getStats()
      .then(stats => sendResponse({ success: true, data: stats }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
});

// Analyze comment using backend API
async function analyzeComment(commentData) {
  const { apiEndpoint } = await chrome.storage.local.get(['apiEndpoint']);
  
  try {
    const response = await fetch(`${apiEndpoint}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(commentData),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Backend analysis failed:', error);
    throw error;
  }
}

// Submit vote to backend
async function submitVote(voteData) {
  const { apiEndpoint } = await chrome.storage.local.get(['apiEndpoint']);
  
  try {
    const response = await fetch(`${apiEndpoint}/vote`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(voteData),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Vote submission failed:', error);
    throw error;
  }
}

// Get statistics
async function getStats() {
  const { votingData } = await chrome.storage.local.get(['votingData']);
  
  if (!votingData) {
    return {
      totalVotes: 0,
      botVotes: 0,
      humanVotes: 0
    };
  }
  
  const votes = Object.values(votingData);
  const botVotes = votes.filter(v => v.vote === 1).length;
  const humanVotes = votes.filter(v => v.vote === -1).length;
  
  return {
    totalVotes: botVotes + humanVotes,
    botVotes,
    humanVotes
  };
}

// Periodic cleanup of old voting data (keep last 30 days)
chrome.alarms.create('cleanupVotes', { periodInMinutes: 1440 }); // Daily

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === 'cleanupVotes') {
    const { votingData } = await chrome.storage.local.get(['votingData']);
    
    if (votingData) {
      const thirtyDaysAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
      const cleaned = {};
      
      for (const [id, data] of Object.entries(votingData)) {
        if (data.timestamp > thirtyDaysAgo) {
          cleaned[id] = data;
        }
      }
      
      await chrome.storage.local.set({ votingData: cleaned });
      console.log(`Cleaned ${Object.keys(votingData).length - Object.keys(cleaned).length} old votes`);
    }
  }
});
