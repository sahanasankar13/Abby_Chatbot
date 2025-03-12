document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');

    // Enable send button when there's text
    userInput.addEventListener('input', function() {
        sendButton.disabled = !userInput.value.trim();
    });

    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = userInput.value.trim();
        if (message) {
            sendMessage(message);
            userInput.value = '';
            sendButton.disabled = true;
        }
    });

    // Initialize chat with a welcome message (without feedback options)
    addBotWelcomeMessage("Hi, I'm Abby. I'm here to provide information about reproductive healthcare and offer support. Everything we discuss is confidential. What questions do you have today?");

    // Functions for chat interaction
    function addUserMessage(message) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';
        messageContainer.style.alignSelf = 'flex-end';

        const messageEl = document.createElement('div');
        messageEl.className = 'message user-message';
        messageEl.textContent = message;

        messageContainer.appendChild(messageEl);
        chatMessages.appendChild(messageContainer);
        scrollToBottom();
    }

    function addBotMessage(message, citations = null, citation_objects = null, graphics = null) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';

        const messageEl = document.createElement('div');
        messageEl.className = 'message bot-message';
        
        // Convert markdown-like syntax to HTML
        let formattedMessage = message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/`(.*?)`/g, '<code>$1</code>') // Code
            .replace(/\n\n/g, '<br><br>') // Line breaks
            .replace(/\n/g, '<br>'); // Line breaks
        
        messageEl.innerHTML = formattedMessage;

        messageContainer.appendChild(messageEl);

        // Add message ID as a data attribute for feedback
        const messageId = 'msg_' + new Date().getTime();
        messageEl.dataset.messageId = messageId;

        // Add citations if provided
        if (citations && citations.length > 0) {
            const citationsContainer = document.createElement('div');
            citationsContainer.className = 'citations-container';
            
            const citationsTitle = document.createElement('div');
            citationsTitle.className = 'citations-title';
            citationsTitle.textContent = 'Sources:';
            
            const citationsList = document.createElement('div');
            citationsList.className = 'citations-list';
            
            citations.forEach(citation => {
                const citationItem = document.createElement('div');
                citationItem.className = 'citation';
                citationItem.innerHTML = citation;
                citationsList.appendChild(citationItem);
            });
            
            citationsContainer.appendChild(citationsTitle);
            citationsContainer.appendChild(citationsList);
            messageContainer.appendChild(citationsContainer);
        }

        // Add graphics if provided
        if (graphics && graphics.length > 0) {
            const graphicsContainer = document.createElement('div');
            graphicsContainer.className = 'graphics-container';
            
            graphics.forEach(graphic => {
                const graphicItem = document.createElement('div');
                graphicItem.className = 'graphic';
                
                const graphicTitle = document.createElement('h4');
                graphicTitle.textContent = graphic.title;
                
                const graphicDesc = document.createElement('div');
                graphicDesc.className = 'graphic-description';
                graphicDesc.textContent = graphic.description;
                
                const svgContainer = document.createElement('div');
                svgContainer.className = 'svg-container';
                svgContainer.innerHTML = graphic.svg;
                
                graphicItem.appendChild(graphicTitle);
                graphicItem.appendChild(graphicDesc);
                graphicItem.appendChild(svgContainer);
                
                graphicsContainer.appendChild(graphicItem);
            });
            
            messageContainer.appendChild(graphicsContainer);
        }

        // Add feedback options
        const feedbackContainer = document.createElement('div');
        feedbackContainer.className = 'feedback-container';
        
        const feedbackLabel = document.createElement('span');
        feedbackLabel.className = 'feedback-label';
        feedbackLabel.textContent = 'Was this helpful?';
        
        const thumbsUp = document.createElement('button');
        thumbsUp.className = 'feedback-btn thumbs-up';
        thumbsUp.innerHTML = '<i class="fas fa-thumbs-up"></i>';
        thumbsUp.addEventListener('click', function() {
            submitFeedback(messageId, 1);
            feedbackContainer.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
        });
        
        const thumbsDown = document.createElement('button');
        thumbsDown.className = 'feedback-btn thumbs-down';
        thumbsDown.innerHTML = '<i class="fas fa-thumbs-down"></i>';
        thumbsDown.addEventListener('click', function() {
            submitFeedback(messageId, -1);
            feedbackContainer.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
        });
        
        feedbackContainer.appendChild(feedbackLabel);
        feedbackContainer.appendChild(thumbsUp);
        feedbackContainer.appendChild(thumbsDown);
        
        messageContainer.appendChild(feedbackContainer);
        
        chatMessages.appendChild(messageContainer);
        scrollToBottom();
    }

    function addTypingIndicator() {
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

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    function addBotWelcomeMessage(message) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';

        const messageEl = document.createElement('div');
        messageEl.className = 'message bot-message';
        
        // Convert markdown-like syntax to HTML
        let formattedMessage = message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/`(.*?)`/g, '<code>$1</code>') // Code
            .replace(/\n\n/g, '<br><br>') // Line breaks
            .replace(/\n/g, '<br>'); // Line breaks
        
        messageEl.innerHTML = formattedMessage;
        messageContainer.appendChild(messageEl);
        
        // No feedback options for welcome message
        
        chatMessages.appendChild(messageContainer);
        scrollToBottom();
    }
    
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function sendMessage(message) {
        addUserMessage(message);
        
        // Check for special command: "end"
        if (message.toLowerCase() === 'end') {
            // Clear history without feedback
            fetch('/api/clear-history', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addBotMessage("Your session has been ended and history has been cleared. If you have more questions in the future, feel free to ask!");
                } else {
                    addBotMessage("I couldn't clear your session history. Please try again.");
                }
            })
            .catch(error => {
                console.error('Error ending session:', error);
                addBotMessage("I'm sorry, there was a problem ending your session. Please try again.");
            });
            return;
        }
        
        // Normal message handling
        addTypingIndicator();
        
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            removeTypingIndicator();
            if (data.error) {
                addBotMessage("I'm sorry, but I encountered an error. Please try again in a moment.");
                console.error('API Error:', data.error);
            } else {
                addBotMessage(
                    data.response, 
                    data.citations, 
                    data.citation_objects || null,
                    data.graphics || null
                );
            }
        })
        .catch(error => {
            removeTypingIndicator();
            console.error('Fetch Error:', error);
            addBotMessage("I'm sorry, but I couldn't connect to the server. Please check your internet connection and try again.");
        });
    }

    function submitFeedback(messageId, rating, comment = null) {
        fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message_id: messageId,
                rating: rating,
                comment: comment
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Feedback submitted:', data);
        })
        .catch(error => {
            console.error('Error submitting feedback:', error);
        });
    }

    // Add ENTER key to send
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendButton.disabled) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // Focus input field on page load
    userInput.focus();
});