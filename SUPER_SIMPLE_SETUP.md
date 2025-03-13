# Super Simple Setup Guide

This guide provides easy step-by-step instructions to get the Reproductive Health Chatbot up and running on your local machine.

## Prerequisites

1. Python 3.9+ installed on your computer
2. API keys:
   - OpenAI API key (get from [OpenAI](https://platform.openai.com/))
   - Abortion Policy API key (get from [Abortion Policy API](https://www.abortionpolicyapi.com/))

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd reproductive-health-chatbot
```

## Step 2: Set Up Environment

### Create and activate a virtual environment

```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Set Up Environment Variables

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key_here
ABORTION_POLICY_API_KEY=your_abortion_policy_api_key_here
SESSION_SECRET=any_random_string_for_sessions
```

## Step 4: Download NLTK Data

```bash
python download_nltk_data.py
```

## Step 5: Run the Application

```bash
python main.py
```

The application will be available at `http://localhost:5000` in your web browser.

## Common Issues

### Issue: ModuleNotFoundError
If you see `ModuleNotFoundError: No module named 'xyz'`, try:
```bash
pip install xyz
```

### Issue: API Key Errors
If you see errors related to API keys:
1. Double-check your `.env` file
2. Make sure the keys are entered correctly with no quotes or extra spaces
3. Verify your OpenAI account has sufficient credits

### Issue: Application crashes with memory error
This application requires significant RAM due to the models. Try:
- Close other memory-intensive applications
- Reduce model complexity in `chatbot/baseline_model.py` by changing `evaluation_model="both"` to `evaluation_model="openai"`

## Need Help?

If you encounter any issues not covered here, please:
1. Check the project's GitHub Issues page
2. Create a new issue with detailed information about your problem