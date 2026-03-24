const chatMessages = document.getElementById('chat-messages') || document.getElementById('chatContainer');
const userInput = document.getElementById('user-input') || document.getElementById('userInput');
const sendButton = document.getElementById('send-button') || document.getElementById('sendBtn');
const exampleButtons = document.querySelectorAll('.example-btn');
const chatStatus = document.getElementById('chat-status');
const HISTORY_LIMIT = 16;
const conversationHistory = [];
const isLegacyLayout = !document.getElementById('chat-messages') && Boolean(document.getElementById('chatContainer'));

function addMessage(message, isUser = false) {
    if (!chatMessages) {
        return;
    }

    const messageDiv = document.createElement('div');
    if (isLegacyLayout) {
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        const speaker = document.createElement('strong');
        speaker.textContent = isUser ? 'You: ' : 'Bobo: ';
        messageDiv.appendChild(speaker);
        messageDiv.appendChild(document.createTextNode(message));
    } else {
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = isUser ? '👤' : '🤖';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = message;

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
    }
    chatMessages.appendChild(messageDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    if (!chatMessages) {
        return;
    }

    const typingDiv = document.createElement('div');
    typingDiv.className = isLegacyLayout ? 'message bot-message' : 'message bot';
    typingDiv.id = 'typing-indicator';
    
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    
    if (!isLegacyLayout) {
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '🤖';
        typingDiv.appendChild(avatar);
    }
    typingDiv.appendChild(indicator);
    chatMessages.appendChild(typingDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

function pushHistory(role, content) {
    conversationHistory.push({ role, content });
    if (conversationHistory.length > HISTORY_LIMIT * 2) {
        conversationHistory.splice(0, conversationHistory.length - HISTORY_LIMIT * 2);
    }
}

async function loadHealthStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        if (chatStatus) {
            if (data.llm_enabled) {
                chatStatus.textContent = `TF-IDF Q&A • SQLite Memory • OpenAI API configured: ${data.llm_model}`;
            } else {
                chatStatus.textContent = 'TF-IDF Q&A • SQLite Memory • Local mode (set OPENAI_API_KEY to enable AI chat)';
            }
        }
        return data;
    } catch (error) {
        if (chatStatus) {
            chatStatus.textContent = 'TF-IDF Q&A • SQLite Memory • Status unavailable';
        }
        return null;
    }
}

async function sendMessage() {
    if (!userInput || !sendButton) {
        return;
    }

    const message = userInput.value.trim();
    if (!message) return;
    
    addMessage(message, true);
    userInput.value = '';
    sendButton.disabled = true;
    
    showTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: conversationHistory.slice(-HISTORY_LIMIT),
            })
        });
        
        const data = await response.json();
        removeTypingIndicator();
        
        if (response.ok && data.response) {
            addMessage(data.response, false);
            pushHistory('user', message);
            pushHistory('assistant', data.response);
        } else {
            addMessage(data.error || 'Sorry, I encountered an error.', false);
        }
    } catch (error) {
        removeTypingIndicator();
        addMessage('Sorry, I could not connect to the server.', false);
        console.error('Error:', error);
    } finally {
        sendButton.disabled = false;
        userInput.focus();
    }
}

if (sendButton) {
    sendButton.addEventListener('click', sendMessage);
}

if (userInput) {
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });
}

exampleButtons.forEach((button) => {
    button.addEventListener('click', () => {
        if (!userInput) {
            return;
        }

        userInput.value = button.dataset.text || '';
        userInput.focus();
    });
});

loadHealthStatus().then((data) => {
    if (data && data.llm_enabled) {
        addMessage(`Hello! I'm your chatbot with ${data.llm_model} configured. If API billing is active, I can answer naturally. How can I help you today?`, false);
        return;
    }
    addMessage('Hello! I\'m your local chatbot. Set OPENAI_API_KEY to enable natural AI conversation.', false);
});
