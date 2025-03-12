document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');

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

            messageContainer.appendChild(messageElement);
            
            // Skip sources completely for short conversational responses
            if (message.length < 100 || !citations || citations.length === 0) {
                chatMessages.appendChild(messageContainer);
                animateMessage(messageElement);
                return;
            }

            // Add citations if present - but only for API or knowledge base sources
            if (citations && citations.length > 0 && citation_objects && citation_objects.length > 0) {
                // Check if any citation is from the abortion policy API or another trusted source
                // Don't show "ai_generated" citations by themselves
                const validSources = ["Abortion Policy API", "Planned Parenthood", "Guttmacher Institute", 
                                     "CDC", "WHO", "American College", "Centers for Disease Control", 
                                     "World Health Organization"];
                
                // Only show citations if we have valid external sources (not just AI-generated)
                const hasApiSources = citation_objects.some(co => {
                    if (!co || !co.source) return false;
                    return validSources.some(validSource => co.source.includes(validSource));
                });

                if (hasApiSources) {
                    const citationsContainer = document.createElement('div');
                    citationsContainer.className = 'citations-container';

                    const citationsTitle = document.createElement('h4');
                    citationsTitle.textContent = 'Sources';

                    const citationsList = document.createElement('div');
                    citationsList.className = 'citations-list';

                    // Filter out AI-generated citations when showing real sources
                    const filteredCitations = citations.filter(c => 
                        !c.includes("AI-generated") && 
                        !c.includes("ai-generated")
                    );

                    filteredCitations.forEach(citation => {
                        const citationElement = document.createElement('div');
                        citationElement.className = 'citation';

                        if (citation.url) {
                            const link = document.createElement('a');
                            link.href = citation.url;
                            link.target = '_blank';
                            link.textContent = citation.title || citation.text || citation.url;
                            citationElement.appendChild(link);
                        } else {
                            citationElement.textContent = citation.text || '';
                        }

                        if (citation.source) {
                            const sourceElement = document.createElement('div');
                            sourceElement.className = 'citation-source';
                            sourceElement.textContent = citation.source;
                            citationElement.appendChild(sourceElement);
                        }

                        citationsList.appendChild(citationElement);
                    });

                    citationsContainer.appendChild(citationsTitle);
                    citationsContainer.appendChild(citationsList);
                    messageContainer.appendChild(citationsContainer);
                }
            }

            // Add graphics if present
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

            // Apply animation
            messageContainer.style.opacity = '0';
            messageContainer.style.transform = 'translateY(10px)';

            chatMessages.appendChild(messageContainer);

            // Trigger animation
            setTimeout(() => {
                messageContainer.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
                messageContainer.style.opacity = '1';
                messageContainer.style.transform = 'translateY(0)';
            }, 10);

            // Scroll to bottom
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

        // Apply animation
        typingIndicator.style.opacity = '0';

        chatMessages.appendChild(typingIndicator);

        // Trigger animation
        setTimeout(() => {
            typingIndicator.style.transition = 'opacity 0.3s ease';
            typingIndicator.style.opacity = '1';
        }, 10);

        scrollToBottom();
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            // Animate out
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

    // Add initial welcome message
    addBotMessage("Hi I'm Abby 👋 I'm here to provide information about reproductive healthcare and offer support. Everything we discuss is private and confidential. Before we begin, please remember not to share any personal details like your name or address - I'm here to help while protecting your privacy. How can I help you today?");

    // Focus input on page load
    userInput.focus();

    // Add Apple-style keyboard handling for mobile
    userInput.addEventListener('focus', () => {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 300);
    });
});