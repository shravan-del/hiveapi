#!/bin/bash
set -e

echo "=== Clearing pip cache ==="
pip cache purge || true

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing base packages ==="
pip install --no-cache-dir Flask==3.0.0 flask-cors==4.0.0 gunicorn==21.2.0
pip install --no-cache-dir praw==7.7.1

echo "=== Installing numpy and pandas (Python 3.13 compatible) ==="
pip install --only-binary :all: numpy==2.2.1
pip install --only-binary :all: pandas==2.2.3

echo "=== Installing OpenAI with httpx ==="
pip uninstall -y openai httpx || true
pip install --no-cache-dir httpx==0.27.2
pip install --no-cache-dir openai==1.54.0

echo "=== Installing ML packages (binary wheels) ==="
pip install --only-binary :all: joblib==1.4.2
pip install --only-binary :all: scipy==1.14.1
pip install --only-binary :all: scikit-learn==1.6.1

echo "=== Installing spaCy (latest available for Python 3.13) ==="
pip install --no-cache-dir spacy==3.8.11

echo "=== Downloading spaCy model ==="
python -m spacy download en_core_web_sm

echo "=== Installing clustering ==="
pip install --no-cache-dir hdbscan==0.8.40

echo "=== Installing sentence transformers ==="
pip install --no-cache-dir sentence-transformers==3.3.1

echo "=== Verifying installations ==="
python -c "import flask; print('Flask OK')"
python -c "import openai; print('OpenAI version:', openai.__version__)"
python -c "import pandas; print('Pandas version:', pandas.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy OK')"
python -c "import hdbscan; print('HDBSCAN OK')"
python -c "import sentence_transformers; print('Sentence Transformers OK')"
python -c "import joblib; print('Joblib OK')"

echo "=== Build complete ==="
