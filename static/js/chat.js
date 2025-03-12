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
            .then(response => response.json())
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
                sendButton.disabled = true;

                // Focus input field again
                userInput.focus();

                // Scroll to the latest message
                scrollToBottom();
            })
            .catch(error => {
                console.error('Error:', error);
                hideTypingIndicator();
                addErrorMessage('Sorry, there was a problem connecting to the server. Please try again.');
                sendButton.disabled = false;
            });
        }
    });

    // Handle input changes to enable/disable send button
    userInput.addEventListener('input', function() {
        sendButton.disabled = userInput.value.trim() === '';
    });

    function addUserMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                ${escapeHTML(message)}
            </div>
            <div class="avatar user-avatar">
                <i class="fas fa-user"></i>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function addBotMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.innerHTML = `
            <div class="avatar bot-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                ${formatMessage(message)}
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function addErrorMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message error-message';
        messageDiv.innerHTML = `
            <div class="avatar bot-avatar">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <div class="message-content">
                ${escapeHTML(message)}
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        chatMessages.appendChild(typingIndicator);
        scrollToBottom();
    }

    function hideTypingIndicator() {
        if (typingIndicator.parentNode === chatMessages) {
            chatMessages.removeChild(typingIndicator);
        }
    }

    function scrollToBottom() {
        // Smooth scroll to bottom with a small delay to ensure rendering is complete
        setTimeout(() => {
            chatMessages.scrollTo({
                top: chatMessages.scrollHeight,
                behavior: 'smooth'
            });
        }, 100);
    }

    function escapeHTML(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatMessage(text) {
        // Convert URLs to clickable links
        let formattedText = text.replace(
            /(https?:\/\/[^\s]+)/g, 
            '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
        );

        // Convert line breaks to <br>
        formattedText = formattedText.replace(/\n/g, '<br>');

        // Format bullet points for better readability
        formattedText = formattedText.replace(/•\s(.*?)(?=(?:•|$))/g, '<div class="bullet-point">• $1</div>');

        return formattedText;
    }

    // Handle viewport height adjustments for mobile
    function setAppHeight() {
        const vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);
    }

    // Set initial height and update on resize
    setAppHeight();
    window.addEventListener('resize', setAppHeight);
});