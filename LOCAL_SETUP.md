# Local Setup Guide

This guide will help you set up and run the Abby Chatbot locally on your machine.

## Prerequisites

### For Mac
- Python 3.11 or higher
- Homebrew (recommended for package management)
- Git
- Terminal application

### For Windows
- Python 3.11 or higher
- Git for Windows
- Command Prompt or PowerShell
- Visual Studio Build Tools (for some package dependencies)

## Setup Instructions

### Mac Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sahanasankar13/Abby_Chatbot.git
   cd Abby_Chatbot
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   cp .env.example .env
   ```
   Open `.env` in your preferred text editor and add your API keys and configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   FLASK_ENV=development
   FLASK_APP=app.py
   SECRET_KEY=your_secret_key_here
   ```

5. **Download NLTK Data**
   ```bash
   python download_nltk_data.py
   ```

6. **Run the Application**
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`

### Windows Setup

1. **Clone the Repository**
   ```cmd
   git clone https://github.com/sahanasankar13/Abby_Chatbot.git
   cd Abby_Chatbot
   ```

2. **Create and Activate Virtual Environment**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```cmd
   copy .env.example .env
   ```
   Open `.env` in your preferred text editor and add your API keys and configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   FLASK_ENV=development
   FLASK_APP=app.py
   SECRET_KEY=your_secret_key_here
   ```

5. **Download NLTK Data**
   ```cmd
   python download_nltk_data.py
   ```

6. **Run the Application**
   ```cmd
   python app.py
   ```
   The application will be available at `http://localhost:5000`

## Troubleshooting

### Common Issues on Mac

1. **Permission Denied**
   ```bash
   chmod +x download_nltk_data.py
   chmod +x app.py
   ```

2. **Python Version Conflict**
   ```bash
   pyenv install 3.11.0
   pyenv global 3.11.0
   ```

3. **SSL Certificate Error**
   ```bash
   pip install --upgrade certifi
   ```

### Common Issues on Windows

1. **Visual C++ Build Tools Error**
   - Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select "Desktop development with C++" during installation

2. **Long Path Issues**
   - Enable long paths in Windows:
     ```cmd
     git config --system core.longpaths true
     ```

3. **Virtual Environment Not Activating**
   - Run PowerShell as Administrator and execute:
     ```powershell
     Set-ExecutionPolicy RemoteSigned
     ```

## Running Tests

```bash
# Mac
python -m pytest

# Windows
python -m pytest
```

## Development Tips

1. **Hot Reload**
   - The development server will automatically reload when you make changes to the code
   - Set `FLASK_ENV=development` in your `.env` file

2. **Debugging**
   - Set `LOG_LEVEL=DEBUG` in your `.env` file for detailed logs
   - Logs are stored in the `logs/` directory

3. **Code Style**
   - Follow PEP 8 guidelines
   - Use a linter (e.g., flake8) for code quality checks

## Next Steps

- Check out the API documentation at `/docs` endpoint
- Visit the admin dashboard at `/admin` endpoint
- Review the metrics at `/admin/metrics` endpoint

## Support

If you encounter any issues:
1. Check the logs in the `logs/` directory
2. Review the error message in the console
3. Create an issue on GitHub with the error details 