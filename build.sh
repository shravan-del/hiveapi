#!/bin/bash
set -e

echo "=== Clearing pip cache ==="
pip cache purge || true

echo "=== Installing setuptools first ==="
pip install --no-cache-dir --upgrade setuptools>=65.0.0

echo "=== Installing base packages ==="
pip install --no-cache-dir Flask==3.0.0 flask-cors==4.0.0 gunicorn==21.2.0
pip install --no-cache-dir praw==7.7.1

echo "=== Installing numpy and pandas (Python 3.13 compatible) ==="
pip install --no-cache-dir numpy==1.26.3
pip install --no-cache-dir pandas==2.2.0

echo "=== Installing OpenAI with httpx ==="
pip uninstall -y openai httpx || true
pip install --no-cache-dir httpx==0.27.2
pip install --no-cache-dir openai==1.54.0

echo "=== Installing ML packages ==="
pip install --no-cache-dir joblib==1.4.0
pip install --no-cache-dir scipy==1.12.0
pip install --no-cache-dir scikit-learn==1.4.0

echo "=== Installing spaCy ==="
pip install --no-cache-dir spacy==3.7.2

echo "=== Downloading spaCy model ==="
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

echo "=== Installing clustering and transformers ==="
pip install --no-cache-dir hdbscan==0.8.33
pip install --no-cache-dir sentence-transformers==2.2.2

echo "=== Verifying installations ==="
python -c "import flask; print('Flask OK')"
python -c "import openai; print('OpenAI version:', openai.__version__)"
python -c "import pandas; print('Pandas version:', pandas.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy OK')"
python -c "import hdbscan; print('HDBSCAN OK')"
python -c "import sentence_transformers; print('Sentence Transformers OK')"
python -c "import joblib; print('Joblib OK')"

echo "=== Build complete ==="
