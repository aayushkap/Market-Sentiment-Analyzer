# Gold Market Sentiment Analyzer

A machine learning powered application that analyzes gold market sentiment from news headlines using an LSTM neural network. The app provides real-time sentiment predictions, trend visualizations, and optional AI-generated sentiment summaries.

## Overview

This project combines a custom-trained LSTM sentiment classifier with an interactive Streamlit dashboard to analyze gold market news. Users can filter data by date range, view prediction accuracy metrics, and optionally enable a local LLM to generate natural language summaries of market trends.

## Dataset

**Source**: [SaguaroCapital/sentiment-analysis-in-commodity-market-gold](https://huggingface.co/datasets/SaguaroCapital/sentiment-analysis-in-commodity-market-gold)

## Model Architecture

The sentiment classifier is a Bidirectional LSTM network:

### Architecture Details: [Notebook](./notebook/LSTM_Sentiment_Analysis.ipynb)

| Layer | Type               | Configuration                        |
| ----- | ------------------ | ------------------------------------ |
| 1     | Input Layer        | Raw text input                       |
| 2     | Embedding Layer    | vocab_size=10,000, embedding_dim=128 |
| 3     | SpatialDropout1D   | rate=0.2                             |
| 4     | Bidirectional LSTM | units=64, return_sequences=True      |
| 5     | Bidirectional LSTM | units=32                             |
| 6     | Dense Layer        | units=64, activation='relu'          |
| 7     | Dropout            | rate=0.5                             |
| 8     | Output Layer       | units=3, activation='softmax'        |

### Model Performance

The trained model achieves strong performance on the test set with balanced accuracy across sentiment classes.

### Optional LLM Summarization

You can enable a lightweight local LLM to generate market sentiment summaries based on the analyzed data. Set `use_llm` in config.ini to True and run the program again. The LLM will automatically download on the next startup.

## Installation & Usage

### Option 1: Docker (Recommended)

1. **Clone the repository**

2. **Build the image**: docker build -t gold-sentiment-analyzer .

3. **Run the container**: docker run -p 8501:8501 gold-sentiment-analyzer

4. **Access the app**: Open your browser to `http://localhost:8501`

### Option 2: Local Python Environment

**Prerequisites**: Python 3.11+

1. **Clone the repository**:

2. **Create virtual environment**: python -m venv venv

3. **Activate virtual environment**: source venv/bin/activate # On Windows: venv\Scripts\activate

4. **Install dependencies**: pip install -r requirements.txt

5. **Run the application**: streamlit run app/main.py

6. **Access the app**: Open your browser to `http://localhost:8501`
