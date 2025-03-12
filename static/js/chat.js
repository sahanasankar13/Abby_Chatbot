document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');
    const sendButton = document.getElementById('sendButton');

    // Add initial bot message
    addBotMessage('Hi there! I\'m your reproductive health assistant. How can I help you today?');

    // Enable input when page loads
    userInput.disabled = false;
    userInput.focus();

    // Check input on every keystroke to enable/disable send button
    userInput.addEventListener('input', function() {
        sendButton.disabled = !userInput.value.trim();
    });

    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const message = userInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addUserMessage(message);

        // Clear input
        userInput.value = '';
        sendButton.disabled = true;

        // Show typing indicator
        showTypingIndicator();
        
        // Send message to server
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add bot response
            if (data.error) {
                addBotMessage("I'm sorry, something went wrong. Please try again later.");
                console.error(data.error);
            } else {
                addBotMessage(formatResponseWithMarkdown(data.response));
            }
        })
        .catch(error => {
            // Hide typing indicator
            hideTypingIndicator();
            
            // Show error message
            addBotMessage("I'm sorry, I couldn't connect to the server. Please try again later.");
            console.error('Error:', error);
        });

        // Send message to server
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            hideTypingIndicator();

            // Format and add bot response to chat
            const formattedResponse = formatResponseWithMarkdown(data.response);
            addBotMessage(formattedResponse);
        })
        .catch(error => {
            console.error('Error:', error);
            hideTypingIndicator();
            addBotMessage('Sorry, I encountered an error. Please try again.');
        })
        .finally(() => {
            // Re-enable input
            userInput.disabled = false;
            userInput.focus();
        });
    });

    // Function to add user message to chat
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.textContent = message;

        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }

    // Function to add bot message to chat
    function addBotMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot-message';
        messageElement.innerHTML = message;

        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }

    // Function to format response with markdown-like syntax
    function formatResponseWithMarkdown(text) {
        // Handle headings
        text = text.replace(/^(#+)\s+(.+)$/gm, function(match, hashes, content) {
            const level = hashes.length;
            return `<h${level}>${content}</h${level}>`;
        });

        // Handle bold
        text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Handle italics
        text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Handle lists
        text = text.replace(/^\s*-\s+(.+)$/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

        // Handle paragraphs
        text = text.replace(/^(.+)$/gm, function(match, content) {
            if (content.startsWith('<h') || content.startsWith('<ul') || content.startsWith('<li') || !content.trim()) {
                return content;
            }
            return `<p>${content}</p>`;
        });

        // Handle new lines
        text = text.replace(/\n\n+/g, '<br>');

        return text;
    }
    
    // Function to add user message to chat
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.textContent = message;
        
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }
    
    // Function to show typing indicator
    function showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.id = 'typingIndicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            typingIndicator.appendChild(dot);
        }
        
        chatMessages.appendChild(typingIndicator);
        scrollToBottom();
    }
    
    // Function to hide typing indicator
    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    // Function to scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Theme toggle functionality
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
            
            // Update icon
            const icon = this.querySelector('i');
            if (icon.classList.contains('fa-moon')) {
                icon.classList.replace('fa-moon', 'fa-sun');
            } else {
                icon.classList.replace('fa-sun', 'fa-moon');
            }
            
            // Save preference
            const isDarkMode = document.body.classList.contains('dark-mode');
            localStorage.setItem('darkMode', isDarkMode);
        });
        
        // Check for saved theme preference
        const savedDarkMode = localStorage.getItem('darkMode') === 'true';
        if (savedDarkMode) {
            document.body.classList.add('dark-mode');
            themeToggle.querySelector('i').classList.replace('fa-moon', 'fa-sun');
        }
    }

    // Function to show typing indicator
    function showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.id = 'typingIndicator';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            typingIndicator.appendChild(dot);
        }

        chatMessages.appendChild(typingIndicator);
        scrollToBottom();
    }

    // Function to hide typing indicator
    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    // Function to scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Add support for Enter key to send message
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendButton.disabled) {
                sendButton.click();
            }
        }
    });
});