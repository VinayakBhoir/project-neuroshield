# NeuroShield: GNN-based Bot Detection

An intelligent Graph Neural Network model for detecting cyber threats (bots, fake accounts) in social networks.

## Project Structure

- `data/` - Dataset storage
- `models/` - GNN model architectures
- `training/` - Training and evaluation logic
- `preprocessing/` - Data cleaning and graph building
- `webapp/` - Streamlit web application
- `configs/` - Configuration files

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (TwiBot-22 or similar)
# Place in data/raw/
```

## execution
python scripts/load_data.py
python scripts/build_graph.py
python scripts/train_model.py

streamlit run webapp/app.py