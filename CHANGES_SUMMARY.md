# Abby Chatbot Bug Fixes (March 2024)

This document summarizes the bug fixes applied to the Abby Chatbot application in March 2024.

## Bug Fixes

### 1. GPT Model Initialization

Fixed multiple issues in the `GPTModel` class initialization:
- Added missing `response_cache` attribute
- Ensured `model_tiers` is properly defined
- Improved error handling for API interactions
- Fixed attribute access errors in the exception handler path

### 2. API Call Parameter Fix

Fixed the call to `record_api_call()` function which was being called with incorrect parameters:
- Changed `record_api_call("openai", model_to_use, total_tokens)` to `record_api_call(f"openai_{model_to_use}", total_tokens)`
- This prevents the system from falling back to the BERT RAG model

### 3. UI Styling Updates

Updated the chat interface to match design requirements:
- Implemented pill-shaped input field with "Ask a question here..." placeholder
- Added an up arrow button for sending messages
- Applied proper styling for the fixed input bar at the bottom of the screen
- Made map display work correctly inline with chat messages

## Files Modified

The following files were updated:
1. `/chatbot/gpt_integration.py`
2. `/chatbot/conversation_manager.py`
3. `/static/css/style.css`
4. `/templates/index.html`
5. `/static/js/maps.js` (new file)

## Deployment Instructions

After pulling these changes, restart your Flask application to apply the fixes. 