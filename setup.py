import subprocess
import sys
import nltk

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

print("Setup complete!")
