# Detailed Local Setup Guide

This guide provides comprehensive instructions for setting up and running the Reproductive Health Chatbot on your local machine. It covers advanced configuration options and troubleshooting tips.

## Prerequisites

- Python 3.9+ (3.11 recommended)
- Git
- pip (Python package installer)
- 8GB+ RAM recommended
- API keys:
  - OpenAI API key with access to GPT-4 models
  - Abortion Policy API key

## 1. Clone the Repository

```bash
git clone <repository-url>
cd reproductive-health-chatbot
```

## 2. Virtual Environment Setup

### Create a virtual environment

```bash
# On Windows
python -m venv venv

# On macOS/Linux
python3 -m venv venv
```

### Activate the virtual environment

```bash
# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Verify activation

Your command prompt should now be prefixed with `(venv)`.

## 3. Install Dependencies

### Install required packages

```bash
pip install -r requirements.txt
```

### Install development dependencies (optional)

```bash
pip install -r requirements-dev.txt
```

## 4. Environment Configuration

### Create a .env file

Copy the example file:

```bash
cp .env.example .env
```

### Edit the .env file

Open the `.env` file in your preferred text editor and add your API keys:

```
# Required environment variables
OPENAI_API_KEY=your_openai_api_key_here
ABORTION_POLICY_API_KEY=your_abortion_policy_api_key_here
SESSION_SECRET=your_random_secret_string_here

# Optional environment variables
# DATABASE_URL=postgresql://user:password@localhost:5432/dbname  # Only needed if using PostgreSQL
# AWS_REGION=us-east-1  # Only needed if using CloudWatch metrics
```

## 5. Data Preparation

### Download NLTK data

```bash
python download_nltk_data.py
```

## 6. Running the Application

### Start the development server

```bash
python main.py
```

The application will be available at http://localhost:5000

### Run with Gunicorn (production-like, Unix/macOS only)

```bash
gunicorn -c gunicorn.conf.py main:app
```

## 7. Advanced Configuration

### Adjusting Model Performance

To reduce memory usage, you can modify the model settings in the conversation manager initialization:

1. Open `app.py`
2. Find the line: `conversation_manager = ConversationManager(evaluation_model="both")`
3. Change it to one of:
   - `evaluation_model="openai"` - Uses only OpenAI for evaluations (lower memory, requires API)
   - `evaluation_model="local"` - Uses only local models (higher memory, no API needed for evaluations)
   - `evaluation_model="both"` - Default, uses both (highest memory usage)

### Customizing Response Behavior

To adjust how the chatbot responds, you can modify the following files:

- `chatbot/friendly_bot.py` - Controls empathetic/friendly tone
- `chatbot/response_evaluator.py` - Adjusts response safety thresholds
- `utils/data_loader.py` - Configures the knowledge base

## 8. Testing

### Run unit tests

```bash
pytest
```

### Test the API connection

```bash
python test_api.py
```

## 9. Common Issues and Solutions

### OpenAI API Key Issues

**Problem**: Error messages containing "OpenAI API key"

**Solutions**:
- Verify your key has been entered correctly in the `.env` file (no quotes, spaces, etc.)
- Check if your OpenAI account has sufficient credits
- Ensure your key has access to the required models (GPT-4 or GPT-3.5-Turbo)

### Memory Issues

**Problem**: Application crashes or becomes unresponsive

**Solutions**:
- Switch to `evaluation_model="openai"` as described in Advanced Configuration
- Close other memory-intensive applications
- Increase your system's swap space
- Run on a machine with more RAM

### Database Connection (if using)

**Problem**: Database connection errors

**Solutions**:
- Verify your `DATABASE_URL` is correctly formatted
- Ensure the database server is running
- Check if the database exists and the user has appropriate permissions

### Module Not Found Errors

**Problem**: `ModuleNotFoundError: No module named 'xyz'`

**Solution**:
```bash
pip install xyz
```

If the module is already in requirements.txt but still not found:
```bash
pip install --force-reinstall -r requirements.txt
```

## 10. Logs and Debugging

### Application Logs

The application logs are saved in `server.log`.

### Setting Debug Level

To change the logging level, modify the `logging.basicConfig` call in `app.py`:

```python
# For more detailed logs
logging.basicConfig(level=logging.DEBUG)

# For fewer logs
logging.basicConfig(level=logging.INFO)
```

## 11. Admin Access

To access the admin dashboard, navigate to `/admin/dashboard` and log in with credentials defined in `users.json`.

## Need More Help?

If you encounter issues not covered in this guide:
1. Check for existing issues in the GitHub repository
2. Open a new issue with detailed information about your problem
3. Contact the development team via the project's communication channels