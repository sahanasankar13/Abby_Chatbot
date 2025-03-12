document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');
    const sendButton = document.getElementById('sendButton');

    // Function to scroll chat to bottom
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to show typing indicator
    function showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'message bot-message typing-indicator';
        typingIndicator.id = 'typingIndicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
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

    // Function to add user message to chat
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.textContent = message;

        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }

    // Function to add bot message to chat with citations and graphics
    function addBotMessage(message, citations = [], graphics = []) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';
        
        // Main message
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot-message';
        messageElement.innerHTML = message;
        messageContainer.appendChild(messageElement);
        
        // Add citations if available
        if (citations && citations.length > 0) {
            const citationsContainer = document.createElement('div');
            citationsContainer.className = 'citations-container';
            
            const citationsTitle = document.createElement('div');
            citationsTitle.className = 'citations-title';
            citationsTitle.textContent = 'Sources:';
            citationsContainer.appendChild(citationsTitle);
            
            const citationsList = document.createElement('div');
            citationsList.className = 'citations-list';
            
            citations.forEach(citation => {
                const citationElement = document.createElement('div');
                citationElement.className = 'citation';
                citationElement.innerHTML = citation;
                citationsList.appendChild(citationElement);
            });
            
            citationsContainer.appendChild(citationsList);
            messageContainer.appendChild(citationsContainer);
        }
        
        // Add graphics if available
        if (graphics && graphics.length > 0) {
            const graphicsContainer = document.createElement('div');
            graphicsContainer.className = 'graphics-container';
            
            graphics.forEach(graphic => {
                if (graphic.type === 'svg') {
                    const graphicElement = document.createElement('div');
                    graphicElement.className = 'graphic';
                    
                    const graphicTitle = document.createElement('h4');
                    graphicTitle.textContent = graphic.title;
                    graphicElement.appendChild(graphicTitle);
                    
                    if (graphic.description) {
                        const graphicDesc = document.createElement('p');
                        graphicDesc.className = 'graphic-description';
                        graphicDesc.textContent = graphic.description;
                        graphicElement.appendChild(graphicDesc);
                    }
                    
                    const svgContainer = document.createElement('div');
                    svgContainer.className = 'svg-container';
                    svgContainer.innerHTML = graphic.content;
                    graphicElement.appendChild(svgContainer);
                    
                    graphicsContainer.appendChild(graphicElement);
                }
            });
            
            messageContainer.appendChild(graphicsContainer);
        }

        chatMessages.appendChild(messageContainer);
        scrollToBottom();
    }

    // Function to format response with markdown-like syntax
    function formatResponseWithMarkdown(text) {
        // First, ensure all numbered lists have consistent formatting
        text = text.replace(/^\s*(\d+)\.\s+(.+)$/gm, '$1. $2');
        
        // Handle headings
        text = text.replace(/^(#+)\s+(.+)$/gm, function(match, hashes, content) {
            const level = Math.min(hashes.length + 2, 6); // h3-h6 to avoid huge headings
            return `<h${level}>${content}</h${level}>`;
        });

        // Special treatment for numbered lists to preserve numbers
        text = text.replace(/^\s*(\d+)\.\s+(.+)$/gm, '<li value="$1">$2</li>');
        
        // Handle bullet lists
        text = text.replace(/^\*\s+(.+)$/gm, '<li>$1</li>');

        // Safely wrap list items in appropriate list tags
        const hasNumberedListItems = text.indexOf('<li value') !== -1;
        const hasBulletListItems = text.match(/<li>(?!value)/);
        
        if (hasNumberedListItems) {
            // Find sequences of numbered list items and wrap them in ol tags
            text = text.replace(/(<li value="[^"]+">.*?<\/li>(\s*<li value="[^"]+">.*?<\/li>)*)/g, '<ol>$1</ol>');
        }
        
        if (hasBulletListItems) {
            // Find sequences of bullet list items and wrap them in ul tags
            text = text.replace(/(<li>(?!value).*?<\/li>(\s*<li>(?!value).*?<\/li>)*)/g, '<ul>$1</ul>');
        }

        // Handle bold text
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Handle italic text
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Better paragraph handling for improved readability
        // First split into paragraphs by double newlines
        const paragraphs = text.split(/\n\n+/);
        
        // Process each paragraph individually
        text = paragraphs.map(para => {
            // Skip if the paragraph is already formatted as a list or header
            if (para.includes('<ol>') || para.includes('<ul>') || 
                para.includes('<h3>') || para.includes('<h4>') || !para.trim()) {
                return para;
            }
            
            // Replace single newlines with <br> within paragraphs
            para = para.replace(/\n/g, '<br>');
            
            // Wrap text in paragraph tag if not already a tag
            if (!para.startsWith('<') || 
                !(para.startsWith('<p>') || para.startsWith('<h') || 
                  para.startsWith('<ul') || para.startsWith('<ol'))) {
                para = '<p>' + para + '</p>';
            }
            
            return para;
        }).join('\n\n');
        
        return text;
    }

    // Add initial bot message
    addBotMessage("Hi I'm Abby! I'm your reproductive health assistant. How can I help you today?");

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

            // Format and add bot response to chat
            const formattedResponse = formatResponseWithMarkdown(data.response);
            addBotMessage(
                formattedResponse, 
                data.citations || [], 
                data.graphics || []
            );
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