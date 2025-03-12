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
                graphicDesc.className = 'graphic-description';raphicDesc.className = 'graphic-description';
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
document.addEventListener('DOMContentLoaded', function() {
    // Chat container and input elements
    const chatContainer = document.querySelector('.messages-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    
    // Theme toggle functionality
    const themeToggle = document.querySelector('.theme-toggle');
    themeToggle.addEventListener('click', function() {
        document.documentElement.setAttribute('data-bs-theme', 
            document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark');
        
        // Update icon
        const icon = this.querySelector('i');
        if (icon.classList.contains('fa-moon')) {
            icon.classList.replace('fa-moon', 'fa-sun');
        } else {
            icon.classList.replace('fa-sun', 'fa-moon');
        }
    });
    
    // Quick exit button (redirect to Google)
    const quickExitBtn = document.querySelector('.quick-exit-btn');
    if (quickExitBtn) {
        quickExitBtn.addEventListener('click', function(e) {
            e.preventDefault();
            window.location.href = 'https://www.google.com';
        });
    }
    
    // End session button
    const endSessionBtn = document.getElementById('end-session');
    if (endSessionBtn) {
        endSessionBtn.addEventListener('click', function() {
            // Clear chat history via API
            fetch('/api/clear-history', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log('History cleared successfully:', data);
                
                // Add end session message
                const botMessage = {
                    text: "Your session has been ended and history has been cleared. If you have more questions in the future, feel free to ask!",
                    sender: 'bot'
                };
                
                // Add the message to UI
                appendMessage(botMessage.text, botMessage.sender);
                
                // Clear the input field
                userInput.value = '';
                
                // Disable input temporarily to prevent sending new messages right after ending
                userInput.disabled = true;
                sendButton.disabled = true;
                
                // Re-enable after a short delay
                setTimeout(() => {
                    userInput.disabled = false;
                    sendButton.disabled = false;
                }, 1500);
            })
            .catch(error => {
                console.error('Error clearing history:', error);
            });
        });
    }
    
    // Send message when button is clicked or Enter is pressed
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Function to send user message and get response
    function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            // Display user message
            appendMessage(message, 'user');
            
            // Clear input field
            userInput.value = '';
            
            // Display typing indicator
            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'typing-indicator';
            typingIndicator.innerHTML = '<span></span><span></span><span></span>';
            chatContainer.appendChild(typingIndicator);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Send request to server
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                // Remove typing indicator
                document.querySelector('.typing-indicator').remove();
                
                // Handle response
                if (data.error) {
                    appendMessage('Sorry, there was an error processing your request. Please try again.', 'bot');
                } else {
                    appendMessage(data.response, 'bot', data.message_id, data.citations, data.graphics);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                // Remove typing indicator
                if (document.querySelector('.typing-indicator')) {
                    document.querySelector('.typing-indicator').remove();
                }
                appendMessage('Sorry, there was an error connecting to the server. Please try again.', 'bot');
            });
        }
    }
    
    // Function to append message to chat
    function appendMessage(text, sender, messageId = null, citations = [], graphics = []) {
        // Create message container
        const messageContainer = document.createElement('div');
        messageContainer.className = `${sender}-message message`;
        
        if (messageId) {
            messageContainer.dataset.messageId = messageId;
        }
        
        // Create message content
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = formatMessage(text);
        messageContainer.appendChild(messageContent);
        
        // Add citations if available
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
        
        // Add feedback options for bot messages
        if (sender === 'bot' && messageId) {
            const feedbackContainer = document.createElement('div');
            feedbackContainer.className = 'feedback-container';
            
            const feedbackText = document.createElement('div');
            feedbackText.className = 'feedback-text';
            feedbackText.textContent = 'Was this helpful?';
            
            const thumbsUp = document.createElement('button');
            thumbsUp.className = 'feedback-button thumbs-up';
            thumbsUp.innerHTML = '<i class="fas fa-thumbs-up"></i>';
            thumbsUp.addEventListener('click', function() {
                submitFeedback(messageId, 1);
                feedbackContainer.innerHTML = '<div class="feedback-thank-you">Thanks for your feedback!</div>';
                setTimeout(() => {
                    feedbackContainer.style.opacity = '0';
                }, 2000);
            });
            
            const thumbsDown = document.createElement('button');
            thumbsDown.className = 'feedback-button thumbs-down';
            thumbsDown.innerHTML = '<i class="fas fa-thumbs-down"></i>';
            thumbsDown.addEventListener('click', function() {
                submitFeedback(messageId, -1);
                feedbackContainer.innerHTML = '<div class="feedback-thank-you">Thanks for your feedback!</div>';
                setTimeout(() => {
                    feedbackContainer.style.opacity = '0';
                }, 2000);
            });
            
            feedbackContainer.appendChild(feedbackText);
            feedbackContainer.appendChild(thumbsUp);
            feedbackContainer.appendChild(thumbsDown);
            messageContainer.appendChild(feedbackContainer);
        }
        
        // Add to chat container
        chatContainer.appendChild(messageContainer);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Function to submit feedback
    function submitFeedback(messageId, rating) {
        fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message_id: messageId,
                rating: rating
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Feedback submitted successfully:', data);
        })
        .catch(error => {
            console.error('Error submitting feedback:', error);
        });
    }
    
    // Function to format message text with markdown-like syntax
    function formatMessage(text) {
        // Replace URLs with clickable links
        let formattedText = text.replace(
            /(https?:\/\/[^\s]+)/g, 
            '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
        );
        
        // Replace line breaks with <br>
        formattedText = formattedText.replace(/\n/g, '<br>');
        
        return formattedText;
    }
});
