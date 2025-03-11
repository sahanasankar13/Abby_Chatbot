document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');
    const sendButton = document.getElementById('sendButton');
    const typingIndicator = document.createElement('div');
    
    // Initialize typing indicator
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = `
        <div class="bubble">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    
    // Add initial welcome message
    addBotMessage("Hi there! I'm your reproductive health assistant. How can I help you today?");
    
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = userInput.value.trim();
        
        if (message) {
            // Add user message to chat
            addUserMessage(message);
            
            // Clear input
            userInput.value = '';
            
            // Show typing indicator
            showTypingIndicator();
            
            // Disable send button while processing
            sendButton.disabled = true;
            
            // Send message to server
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Hide typing indicator
                hideTypingIndicator();
                
                // Add bot response to chat
                if (data.error) {
                    addErrorMessage(data.error);
                } else {
                    addBotMessage(data.response);
                }
                
                // Re-enable send button
                sendButton.disabled = false;
                
                // Scroll to bottom
                scrollToBottom();
            })
            .catch(error => {
                console.error('Error:', error);
                hideTypingIndicator();
                addErrorMessage("Sorry, I couldn't process your message. Please try again.");
                sendButton.disabled = false;
                scrollToBottom();
            });
        }
    });
    
    // Handle input changes to enable/disable send button
    userInput.addEventListener('input', function() {
        sendButton.disabled = userInput.value.trim() === '';
    });
    
    // Function to add user message to chat
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.innerHTML = `
            <div class="message-content">
                <p>${escapeHtml(message)}</p>
            </div>
        `;
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }
    
    // Function to add bot message to chat
    function addBotMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot-message';
        messageElement.innerHTML = `
            <div class="avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <p>${formatMessage(message)}</p>
            </div>
        `;
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }
    
    // Function to add error message to chat
    function addErrorMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message error-message';
        messageElement.innerHTML = `
            <div class="avatar">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <div class="message-content">
                <p>${escapeHtml(message)}</p>
            </div>
        `;
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }
    
    // Function to show typing indicator
    function showTypingIndicator() {
        chatMessages.appendChild(typingIndicator);
        scrollToBottom();
    }
    
    // Function to hide typing indicator
    function hideTypingIndicator() {
        if (typingIndicator.parentNode === chatMessages) {
            chatMessages.removeChild(typingIndicator);
        }
    }
    
    // Function to scroll chat to bottom
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to escape HTML to prevent XSS
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Function to format message with markdown-like syntax
    function formatMessage(message) {
        // Replace newlines with <br>
        let formatted = message.replace(/\n/g, '<br>');
        
        // Bold text between ** **
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic text between * *
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Links
        formatted = formatted.replace(
            /\[(.*?)\]\((https?:\/\/[^\s]+)\)/g, 
            '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
        );
        
        return formatted;
    }
});
