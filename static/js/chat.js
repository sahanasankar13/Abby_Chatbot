document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const suggestedPrompts = document.getElementById('suggestedPrompts');
    const promptButtons = document.querySelectorAll('.prompt-btn');

    // Enable send button when there's text
    userInput.addEventListener('input', function() {
        sendButton.disabled = !userInput.value.trim();
        
        // Hide suggested prompts when user starts typing
        if (userInput.value.trim()) {
            const suggestedPrompts = document.getElementById('suggestedPrompts');
            if (suggestedPrompts) {
                suggestedPrompts.style.display = 'none';
            }
        }
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
    
    // Handle suggestion prompt buttons
    promptButtons.forEach(button => {
        button.addEventListener('click', function() {
            const promptText = this.getAttribute('data-prompt');
            userInput.value = promptText;
            sendButton.disabled = false;
            // Focus on input so user can modify if desired
            userInput.focus();
            
            // Auto-send after a short delay if user doesn't modify
            setTimeout(() => {
                if (userInput.value === promptText) {
                    sendMessage(promptText);
                    userInput.value = '';
                    sendButton.disabled = true;
                }
            }, 800);
        });
    });

    // Initialize chat with a welcome message (without feedback options)
    addBotWelcomeMessage("Hi! 👋 How can I help you today?");

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
        
        // Add suggestion prompts after welcome message
        const suggestedPrompts = document.createElement('div');
        suggestedPrompts.id = 'suggestedPrompts';
        suggestedPrompts.className = 'suggested-prompts';
        
        suggestedPrompts.innerHTML = `
            <div class="prompts-container">
                <div class="prompt-row">
                    <button class="prompt-btn" data-prompt="Can I get an abortion in my state?">Can I get an abortion in my state? 🗺️</button>
                    <button class="prompt-btn" data-prompt="What contraception methods are available?">What contraception methods are available? 💊</button>
                </div>
                <div class="prompt-row">
                    <button class="prompt-btn" data-prompt="How does pregnancy happen?">How does pregnancy happen? 🤰</button>
                    <button class="prompt-btn" data-prompt="What are some stress management tips?">What are some stress management tips? 🧘</button>
                </div>
                <div class="prompt-row">
                    <button class="prompt-btn" data-prompt="Explain STI prevention">Explain STI prevention 🛡️</button>
                    <button class="prompt-btn" data-prompt="What are the signs of pregnancy?">What are the signs of pregnancy? 🔍</button>
                </div>
            </div>
        `;
        
        chatMessages.appendChild(suggestedPrompts);
        
        // Add event listeners to the prompt buttons
        const promptButtons = suggestedPrompts.querySelectorAll('.prompt-btn');
        promptButtons.forEach(button => {
            button.addEventListener('click', function() {
                const promptText = this.getAttribute('data-prompt');
                userInput.value = promptText;
                sendButton.disabled = false;
                // Focus on input so user can modify if desired
                userInput.focus();
                
                // Hide suggested prompts immediately when clicked
                const suggestedPrompts = document.getElementById('suggestedPrompts');
                if (suggestedPrompts) {
                    suggestedPrompts.style.display = 'none';
                }
                
                // Auto-send after a short delay if user doesn't modify
                setTimeout(() => {
                    if (userInput.value === promptText) {
                        sendMessage(promptText);
                        userInput.value = '';
                        sendButton.disabled = true;
                    }
                }, 800);
            });
        });
        
        scrollToBottom();
    }
    
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function sendMessage(message) {
        addUserMessage(message);
        
        // Hide suggested prompts when user sends a message
        const suggestedPrompts = document.getElementById('suggestedPrompts');
        if (suggestedPrompts) {
            suggestedPrompts.style.display = 'none';
        }
        
        // Check for special command: "end"
        if (message.toLowerCase() === 'end') {
            // Show feedback dialog before ending session
            const feedbackDialog = document.createElement('div');
            feedbackDialog.className = 'feedback-dialog';
            feedbackDialog.id = 'end-session-feedback';
            
            const dialogContent = document.createElement('div');
            dialogContent.className = 'feedback-dialog-content';
            
            const title = document.createElement('h3');
            title.textContent = 'Before you go...';
            
            const text = document.createElement('p');
            text.textContent = 'Would you like to provide feedback on your experience?';
            
            const buttonsContainer = document.createElement('div');
            buttonsContainer.className = 'feedback-dialog-buttons';
            
            const provideFeedbackBtn = document.createElement('button');
            provideFeedbackBtn.className = 'provide-feedback-btn';
            provideFeedbackBtn.textContent = 'Provide Feedback';
            provideFeedbackBtn.addEventListener('click', function() {
                // Remove current dialog
                feedbackDialog.remove();
                
                // Show feedback form
                showFeedbackForm();
            });
            
            const skipFeedbackBtn = document.createElement('button');
            skipFeedbackBtn.className = 'skip-feedback-btn';
            skipFeedbackBtn.textContent = 'No Thanks';
            skipFeedbackBtn.addEventListener('click', function() {
                feedbackDialog.remove();
                clearSessionHistory();
            });
            
            buttonsContainer.appendChild(provideFeedbackBtn);
            buttonsContainer.appendChild(skipFeedbackBtn);
            
            dialogContent.appendChild(title);
            dialogContent.appendChild(text);
            dialogContent.appendChild(buttonsContainer);
            
            feedbackDialog.appendChild(dialogContent);
            document.body.appendChild(feedbackDialog);
            
            return;
        }
        
        // Helper function to show feedback form 
        function showFeedbackForm() {
            const feedbackForm = document.createElement('div');
            feedbackForm.className = 'feedback-dialog';
            feedbackForm.id = 'feedback-form-dialog';
            
            const formContent = document.createElement('div');
            formContent.className = 'feedback-dialog-content';
            
            const title = document.createElement('h3');
            title.textContent = 'Your Feedback';
            
            const ratingContainer = document.createElement('div');
            ratingContainer.className = 'rating-container';
            
            const ratingLabel = document.createElement('label');
            ratingLabel.textContent = 'How would you rate your experience?';
            ratingLabel.className = 'rating-label';
            
            const starsContainer = document.createElement('div');
            starsContainer.className = 'stars-container';
            
            for (let i = 1; i <= 5; i++) {
                const star = document.createElement('span');
                star.className = 'feedback-star';
                star.innerHTML = '★';
                star.dataset.value = i;
                star.addEventListener('click', function() {
                    // Clear all previously selected stars
                    const allStars = star.parentNode.querySelectorAll('.feedback-star');
                    allStars.forEach(s => s.classList.remove('selected'));
                    
                    // Select this star and all stars before it
                    for (let j = 0; j < i; j++) {
                        allStars[j].classList.add('selected');
                    }
                    star.classList.add('selected');
                });
                starsContainer.appendChild(star);
            }
            
            const commentContainer = document.createElement('div');
            commentContainer.className = 'comment-container';
            
            const commentLabel = document.createElement('label');
            commentLabel.textContent = 'Any additional comments?';
            commentLabel.className = 'comment-label';
            
            const commentInput = document.createElement('textarea');
            commentInput.className = 'feedback-comment';
            commentInput.placeholder = 'Your comments help us improve...';
            
            const buttonsContainer = document.createElement('div');
            buttonsContainer.className = 'feedback-dialog-buttons';
            
            const submitBtn = document.createElement('button');
            submitBtn.className = 'provide-feedback-btn';
            submitBtn.textContent = 'Submit Feedback';
            submitBtn.addEventListener('click', function() {
                // Get selected rating
                const selectedStars = document.querySelectorAll('.feedback-star.selected');
                const rating = selectedStars.length; // Number of selected stars is the rating
                const comment = commentInput.value.trim();
                
                // Submit feedback
                fetch('/api/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message_id: 'end_session_' + Date.now(),
                        rating: rating,
                        comment: comment
                    })
                })
                .then(response => response.json())
                .catch(error => {
                    console.error('Error submitting feedback:', error);
                });
                
                // Close form and clear history
                feedbackForm.remove();
                clearSessionHistory();
            });
            
            const skipBtn = document.createElement('button');
            skipBtn.className = 'skip-feedback-btn';
            skipBtn.textContent = 'Skip';
            skipBtn.addEventListener('click', function() {
                feedbackForm.remove();
                clearSessionHistory();
            });
            
            ratingContainer.appendChild(ratingLabel);
            ratingContainer.appendChild(starsContainer);
            
            commentContainer.appendChild(commentLabel);
            commentContainer.appendChild(commentInput);
            
            buttonsContainer.appendChild(skipBtn);
            buttonsContainer.appendChild(submitBtn);
            
            formContent.appendChild(title);
            formContent.appendChild(ratingContainer);
            formContent.appendChild(commentContainer);
            formContent.appendChild(buttonsContainer);
            
            feedbackForm.appendChild(formContent);
            document.body.appendChild(feedbackForm);
        }
        
        // Helper function to reset the UI for a new session
        function clearSessionHistory() {
            // Mark session as complete on server (preserves logs/history for analytics)
            fetch('/api/clear-history', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Clear all messages from the UI only
                    chatMessages.innerHTML = '';
                    
                    // Re-add the welcome message with suggested prompts
                    addBotWelcomeMessage("Hi! 👋 How can I help you today?");
                    
                    // Add confirmation message
                    addBotMessage("Your session has been ended. If you have more questions in the future, feel free to ask!");
                } else {
                    addBotMessage("I couldn't end your session. Please try again.");
                }
            })
            .catch(error => {
                console.error('Error ending session:', error);
                addBotMessage("I'm sorry, there was a problem ending your session. Please try again.");
            });
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
