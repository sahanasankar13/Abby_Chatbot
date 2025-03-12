document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    
    // Add end session button to the header
    const chatHeader = document.querySelector('.chat-header');
    const endSessionBtn = document.createElement('button');
    endSessionBtn.className = 'end-session-btn';
    endSessionBtn.innerHTML = '<i class="fas fa-sign-out-alt"></i> End Session';
    endSessionBtn.addEventListener('click', endSession);
    chatHeader.appendChild(endSessionBtn);
    
    // Add optional feedback button
    const feedbackBtn = document.createElement('button');
    feedbackBtn.className = 'feedback-btn-header';
    feedbackBtn.innerHTML = '<i class="fas fa-comment-alt"></i> Give Feedback';
    feedbackBtn.addEventListener('click', openFeedbackModal);
    chatHeader.appendChild(feedbackBtn);

    // Enable/disable send button based on input
    userInput.addEventListener('input', function() {
        sendButton.disabled = userInput.value.trim() === '';
    });

    // Handle form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const message = userInput.value.trim();
        if (message === '') return;

        // Add user message to chat
        addUserMessage(message);

        // Clear input
        userInput.value = '';
        sendButton.disabled = true;

        // Show typing indicator
        addTypingIndicator();

        // Send message to backend
        sendMessage(message);
    });

    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.textContent = message;

        // Apply animation class
        messageElement.style.opacity = '0';
        messageElement.style.transform = 'translateY(10px)';

        chatMessages.appendChild(messageElement);

        // Trigger animation
        setTimeout(() => {
            messageElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            messageElement.style.opacity = '1';
            messageElement.style.transform = 'translateY(0)';
        }, 10);

        // Scroll to bottom
        scrollToBottom();
    }

    function addBotMessage(message, citations = null, citation_objects = null, graphics = null) {
        try {
            // Remove typing indicator first
            removeTypingIndicator();

            const messageContainer = document.createElement('div');
            messageContainer.className = 'message-container';

            const messageElement = document.createElement('div');
            messageElement.className = 'message bot-message';
            messageElement.innerHTML = message;
            
            // Generate a unique message ID if needed
            const messageId = 'msg_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
            messageElement.dataset.messageId = messageId;

            messageContainer.appendChild(messageElement);
            
            // Add feedback buttons
            const feedbackContainer = document.createElement('div');
            feedbackContainer.className = 'feedback-container';
            
            const feedbackLabel = document.createElement('span');
            feedbackLabel.className = 'feedback-label';
            feedbackLabel.textContent = 'Was this helpful?';
            feedbackContainer.appendChild(feedbackLabel);
            
            const thumbsUpBtn = document.createElement('button');
            thumbsUpBtn.className = 'feedback-btn thumbs-up';
            thumbsUpBtn.innerHTML = '<i class="fas fa-thumbs-up"></i>';
            thumbsUpBtn.title = 'This was helpful';
            thumbsUpBtn.addEventListener('click', function() {
                submitFeedback(messageId, 1);
                feedbackContainer.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
            });
            feedbackContainer.appendChild(thumbsUpBtn);
            
            const thumbsDownBtn = document.createElement('button');
            thumbsDownBtn.className = 'feedback-btn thumbs-down';
            thumbsDownBtn.innerHTML = '<i class="fas fa-thumbs-down"></i>';
            thumbsDownBtn.title = 'This was not helpful';
            thumbsDownBtn.addEventListener('click', function() {
                submitFeedback(messageId, -1);
                feedbackContainer.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
            });
            feedbackContainer.appendChild(thumbsDownBtn);
            
            messageContainer.appendChild(feedbackContainer);


            if ((message.length < 100 && !message.includes("abortion")) ||
                !citations || !Array.isArray(citations) || citations.length === 0) {
                chatMessages.appendChild(messageContainer);
                messageContainer.style.opacity = '0';
                messageContainer.style.transform = 'translateY(10px)';

                setTimeout(() => {
                    messageContainer.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
                    messageContainer.style.opacity = '1';
                    messageContainer.style.transform = 'translateY(0)';
                }, 10);
                return;
            }

            const safeObjects = (citation_objects && Array.isArray(citation_objects)) ? citation_objects : [];

            // Add citations if present - but only for API or knowledge base sources
            if (citations && Array.isArray(citations) && citations.length > 0) {
                const validSources = ["Abortion Policy API", "Planned Parenthood", "Guttmacher Institute",
                                     "CDC", "WHO", "American College", "Centers for Disease Control",
                                     "World Health Organization"];

                const hasApiSources = safeObjects.some(co => {
                    if (!co || !co.source) return false;
                    return validSources.some(validSource =>
                        co.source && typeof co.source === 'string' && co.source.includes(validSource)
                    );
                });

                const hasPolicyContent = message.includes("policy") || message.includes("state") ||
                                        message.includes("law") || message.includes("abortion");

                if (hasApiSources || hasPolicyContent || citations.length > 0) {
                    const citationsContainer = document.createElement('div');
                    citationsContainer.className = 'citations-container';

                    const citationsTitle = document.createElement('h4');
                    citationsTitle.textContent = 'Sources';

                    const citationsList = document.createElement('div');
                    citationsList.className = 'citations-list';


                    let filteredCitations = [];
                    try {
                        filteredCitations = citations.filter(c => {
                            if (typeof c !== 'string' && typeof c !== 'object') return false;
                            if (typeof c === 'string') {
                                return !c.includes("AI-generated") && !c.includes("ai-generated");
                            }
                            return true;
                        });
                    } catch (filterError) {
                        console.error("Error filtering citations:", filterError);
                        filteredCitations = Array.isArray(citations) ? [...citations] : [];
                    }

                    if (filteredCitations.length === 0 && citations.length > 0) {
                        filteredCitations.push({
                            source: "Planned Parenthood",
                            url: "https://www.plannedparenthood.org/",
                            title: "Planned Parenthood"
                        });
                    }

                    filteredCitations.forEach(citation => {
                        try {
                            const citationElement = document.createElement('div');
                            citationElement.className = 'citation';

                            if (typeof citation === 'string' && citation.startsWith('<div class="citation">')) {
                                citationElement.innerHTML = citation;
                            }
                            else if (citation && typeof citation === 'object') {
                                if (citation.url) {
                                    const link = document.createElement('a');
                                    link.href = citation.url;
                                    link.target = '_blank';
                                    link.textContent = citation.title || citation.text || citation.url;
                                    citationElement.appendChild(link);
                                } else {
                                    citationElement.textContent = citation.text || citation.source || 'Citation';
                                }

                                if (citation.source) {
                                    const sourceElement = document.createElement('div');
                                    sourceElement.className = 'citation-source';
                                    sourceElement.textContent = citation.source;
                                    citationElement.appendChild(sourceElement);
                                }
                            }
                            else if (typeof citation === 'string') {
                                citationElement.textContent = citation;
                            }
                            else {
                                citationElement.textContent = "Source information available upon request";
                            }

                            citationsList.appendChild(citationElement);
                        } catch (citationError) {
                            console.error("Error processing citation:", citationError, citation);
                        }
                    });

                    if (citationsList.children.length > 0) {
                        citationsContainer.appendChild(citationsTitle);
                        citationsContainer.appendChild(citationsList);
                        messageContainer.appendChild(citationsContainer);
                    }
                }
            }


            if (graphics && graphics.length > 0) {
                const graphicsContainer = document.createElement('div');
                graphicsContainer.className = 'graphics-container';

                graphics.forEach(graphic => {
                    const graphicElement = document.createElement('div');
                    graphicElement.className = 'graphic';

                    const title = document.createElement('h4');
                    title.textContent = graphic.title || 'Visual Information';
                    graphicElement.appendChild(title);

                    if (graphic.description) {
                        const description = document.createElement('div');
                        description.className = 'graphic-description';
                        description.textContent = graphic.description;
                        graphicElement.appendChild(description);
                    }

                    const svgContainer = document.createElement('div');
                    svgContainer.className = 'svg-container';
                    svgContainer.innerHTML = graphic.svg;
                    graphicElement.appendChild(svgContainer);

                    graphicsContainer.appendChild(graphicElement);
                });

                messageContainer.appendChild(graphicsContainer);
            }

            messageContainer.style.opacity = '0';
            messageContainer.style.transform = 'translateY(10px)';

            chatMessages.appendChild(messageContainer);

            setTimeout(() => {
                messageContainer.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
                messageContainer.style.opacity = '1';
                messageContainer.style.transform = 'translateY(0)';
            }, 10);

            scrollToBottom();
        } catch (error) {
            console.log("Error adding bot message:", error);
        }
    }

    function addTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.id = 'typingIndicator';
        typingIndicator.className = 'typing-indicator';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            typingIndicator.appendChild(dot);
        }

        typingIndicator.style.opacity = '0';

        chatMessages.appendChild(typingIndicator);

        setTimeout(() => {
            typingIndicator.style.transition = 'opacity 0.3s ease';
            typingIndicator.style.opacity = '1';
        }, 10);

        scrollToBottom();
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.style.opacity = '0';

            setTimeout(() => {
                typingIndicator.remove();
            }, 300);
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function sendMessage(message) {
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            addBotMessage(
                data.response,
                data.citations || [],
                data.citation_objects || [],
                data.graphics || []
            );
        })
        .catch(error => {
            removeTypingIndicator();
            console.error("Error:", error);
            addBotMessage("I'm sorry, but I wasn't able to process your request. Please try again later.");
        });
    }

    // Add initial welcome message - without feedback buttons
    const welcomeMessage = "Hi I'm Abby 👋 I'm here to provide information about reproductive healthcare and offer support. Everything we discuss is private and confidential. Before we begin, please remember not to share any personal details like your name or address - I'm here to help while protecting your privacy. How can I help you today?";
    
    // Special handling for the welcome message - no feedback buttons
    const welcomeContainer = document.createElement('div');
    welcomeContainer.className = 'message-container';
    
    const messageElement = document.createElement('div');
    messageElement.className = 'message bot-message';
    messageElement.innerHTML = welcomeMessage;
    
    welcomeContainer.appendChild(messageElement);
    welcomeContainer.style.opacity = '0';
    welcomeContainer.style.transform = 'translateY(10px)';
    
    chatMessages.appendChild(welcomeContainer);
    
    setTimeout(() => {
        welcomeContainer.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
        welcomeContainer.style.opacity = '1';
        welcomeContainer.style.transform = 'translateY(0)';
    }, 10);

    userInput.focus();

    userInput.addEventListener('focus', () => {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 300);
    });
    
    // Feedback handling functions
    function submitFeedback(messageId, rating, comment = null) {
        fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message_id: messageId,
                rating: rating,
                comment: comment
            }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Feedback submitted successfully:', data);
            // Track metrics for feedback
            if (typeof window.trackMetric === 'function') {
                window.trackMetric('feedback_submitted', { rating: rating });
            }
        })
        .catch(error => {
            console.error('Error submitting feedback:', error);
        });
    }
    
    function openFeedbackModal() {
        // Create modal overlay
        const modalOverlay = document.createElement('div');
        modalOverlay.className = 'modal-overlay';
        
        // Create modal container
        const modalContainer = document.createElement('div');
        modalContainer.className = 'feedback-modal';
        
        // Create modal content
        modalContainer.innerHTML = `
            <div class="modal-header">
                <h3>Session Feedback</h3>
                <button class="close-modal">&times;</button>
            </div>
            <div class="modal-body">
                <p>Thank you for using Abby. Your feedback helps us improve.</p>
                <div class="feedback-rating">
                    <p>How would you rate your overall experience?</p>
                    <div class="rating-buttons">
                        <button class="rating-btn" data-rating="5">Excellent</button>
                        <button class="rating-btn" data-rating="4">Good</button>
                        <button class="rating-btn" data-rating="3">Okay</button>
                        <button class="rating-btn" data-rating="2">Poor</button>
                        <button class="rating-btn" data-rating="1">Bad</button>
                    </div>
                </div>
                <div class="feedback-comment">
                    <p>Do you have any additional comments or suggestions?</p>
                    <textarea id="feedbackComment" placeholder="Your feedback helps us improve (optional)"></textarea>
                </div>
            </div>
            <div class="modal-footer">
                <button class="submit-feedback-btn" disabled>Submit Feedback</button>
                <button class="skip-feedback-btn">Skip & Close</button>
            </div>
        `;
        
        // Add modal to body
        modalOverlay.appendChild(modalContainer);
        document.body.appendChild(modalOverlay);
        
        // Handle close button
        const closeBtn = modalContainer.querySelector('.close-modal');
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(modalOverlay);
        });
        
        // Handle rating selection
        const ratingBtns = modalContainer.querySelectorAll('.rating-btn');
        const submitBtn = modalContainer.querySelector('.submit-feedback-btn');
        let selectedRating = null;
        
        ratingBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Clear previous selection
                ratingBtns.forEach(b => b.classList.remove('selected'));
                // Add selected class to clicked button
                btn.classList.add('selected');
                // Store rating value
                selectedRating = parseInt(btn.dataset.rating);
                // Enable submit button
                submitBtn.disabled = false;
            });
        });
        
        // Handle submit button
        submitBtn.addEventListener('click', () => {
            if (selectedRating !== null) {
                const comment = document.getElementById('feedbackComment').value.trim();
                // Convert rating scale (1-5) to thumbs format (1 for positive, -1 for negative)
                const thumbsRating = selectedRating >= 3 ? 1 : -1;
                
                // Submit session feedback with a special message ID for session feedback
                submitFeedback('session_feedback_' + Date.now(), thumbsRating, comment);
                
                // Thank user and reset chat
                document.body.removeChild(modalOverlay);
                resetChat();
            }
        });
    }
    
    function endSession() {
        // Add end session message
        addBotMessage("Your session has been ended. Thank you for using Abby! If you have more questions, feel free to start a new conversation.");
        
        // Reset the chat after a brief delay
        setTimeout(() => {
            // Clear all messages
            while (chatMessages.children.length > 0) {
                chatMessages.removeChild(chatMessages.lastChild);
            }
            
            // Add welcome message back with special handling
            const welcomeMessage = "Hi I'm Abby 👋 I'm here to provide information about reproductive healthcare and offer support. Everything we discuss is private and confidential. Before we begin, please remember not to share any personal details like your name or address - I'm here to help while protecting your privacy. How can I help you today?";
            
            const welcomeContainer = document.createElement('div');
            welcomeContainer.className = 'message-container';
            
            const messageElement = document.createElement('div');
            messageElement.className = 'message bot-message';
            messageElement.innerHTML = welcomeMessage;
            
            welcomeContainer.appendChild(messageElement);
            chatMessages.appendChild(welcomeContainer);
        }, 2000);
    }
    
    function resetChat() {
        // Clear chat messages except for the first welcome message
        while (chatMessages.children.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
        
        // If all messages were removed, add welcome message back
        if (chatMessages.children.length === 0) {
            // Add welcome message with special handling (no feedback buttons)
            const welcomeMessage = "Hi I'm Abby 👋 I'm here to provide information about reproductive healthcare and offer support. Everything we discuss is private and confidential. Before we begin, please remember not to share any personal details like your name or address - I'm here to help while protecting your privacy. How can I help you today?";
            
            const welcomeContainer = document.createElement('div');
            welcomeContainer.className = 'message-container';
            
            const messageElement = document.createElement('div');
            messageElement.className = 'message bot-message';
            messageElement.innerHTML = welcomeMessage;
            
            welcomeContainer.appendChild(messageElement);
            chatMessages.appendChild(welcomeContainer);
        }
        
        // Show session ended message
        addBotMessage("Your session has been ended and your feedback has been submitted. Thank you for using Abby! If you have more questions, feel free to start a new conversation.");
    }
});