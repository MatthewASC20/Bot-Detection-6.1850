// Popup script for YouTube Bot Detector settings

document.addEventListener('DOMContentLoaded', async () => {
  // Load current settings and stats
  await loadSettings();
  await loadStats();
  
  // Set up event listeners
  setupEventListeners();
});

async function loadSettings() {
  const settings = await chrome.storage.local.get([
    'apiEndpoint',
    'detectionEnabled',
    'votingEnabled'
  ]);
  
  document.getElementById('apiEndpoint').value = settings.apiEndpoint || 'http://localhost:5000/api';
  document.getElementById('detectionToggle').checked = settings.detectionEnabled !== false;
  document.getElementById('votingToggle').checked = settings.votingEnabled !== false;
}

async function loadStats() {
  try {
    const { votingData } = await chrome.storage.local.get(['votingData']);
    
    if (!votingData) {
      return;
    }
    
    const votes = Object.values(votingData);
    const botVotes = votes.filter(v => v.vote === 1).length;
    const humanVotes = votes.filter(v => v.vote === -1).length;
    const totalVotes = botVotes + humanVotes;
    
    document.getElementById('totalVotes').textContent = totalVotes;
    document.getElementById('botVotes').textContent = botVotes;
    document.getElementById('humanVotes').textContent = humanVotes;
  } catch (error) {
    console.error('Error loading stats:', error);
  }
}

function setupEventListeners() {
  // Test connection button
  document.getElementById('testConnection').addEventListener('click', async () => {
    const apiEndpoint = document.getElementById('apiEndpoint').value;
    await testConnection(apiEndpoint);
  });
  
  // Save settings button
  document.getElementById('saveSettings').addEventListener('click', async () => {
    await saveSettings();
  });
  
  // Clear data button
  document.getElementById('clearData').addEventListener('click', async () => {
    if (confirm('Are you sure you want to clear all voting data? This cannot be undone.')) {
      await clearAllData();
    }
  });
  
  // Toggle switches
  document.getElementById('detectionToggle').addEventListener('change', async (e) => {
    await chrome.storage.local.set({ detectionEnabled: e.target.checked });
    showStatus('Detection ' + (e.target.checked ? 'enabled' : 'disabled'), 'success');
  });
  
  document.getElementById('votingToggle').addEventListener('change', async (e) => {
    await chrome.storage.local.set({ votingEnabled: e.target.checked });
    showStatus('Voting ' + (e.target.checked ? 'enabled' : 'disabled'), 'success');
  });
}

async function testConnection(apiEndpoint) {
  showStatus('Testing connection...', 'success');
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch(`${apiEndpoint}/health`, {
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      showStatus('✓ Connection successful!', 'success');
    } else {
      showStatus('✗ Server responded with error: ' + response.status, 'error');
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      showStatus('✗ Connection timeout. Is the backend running?', 'error');
    } else {
      showStatus('✗ Connection failed: ' + error.message, 'error');
    }
  }
}

async function saveSettings() {
  const apiEndpoint = document.getElementById('apiEndpoint').value;
  const detectionEnabled = document.getElementById('detectionToggle').checked;
  const votingEnabled = document.getElementById('votingToggle').checked;
  
  await chrome.storage.local.set({
    apiEndpoint,
    detectionEnabled,
    votingEnabled
  });
  
  showStatus('✓ Settings saved successfully!', 'success');
}

async function clearAllData() {
  await chrome.storage.local.set({ votingData: {} });
  await loadStats();
  showStatus('✓ All voting data cleared', 'success');
}

function showStatus(message, type) {
  const statusElement = document.getElementById('statusMessage');
  statusElement.textContent = message;
  statusElement.className = 'status ' + type;
  statusElement.style.display = 'block';
  
  setTimeout(() => {
    statusElement.style.display = 'none';
  }, 3000);
}
