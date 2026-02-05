# NeuroShield: GNN-based Bot Detection

An intelligent Graph Neural Network model for detecting cyber threats (bots, fake accounts) in social networks.



## Project Structure

- `data/` - Dataset storage
- `models/` - GNN model architectures
- `training/` - Training and evaluation logic
- `preprocessing/` - Data cleaning and graph building
- `webapp/` - Streamlit web application
- `configs/` - Configuration files

---

##  Setup

### Create a Virtual Environment
```bash
python -m venv venv 
venv \Scripts\activate
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

###  Dataset Setup

- Download the dataset (cresci) [link](https://drive.google.com/drive/folders/15j6JxduklP9MXiH2pz-YXfp6V8Pliw4r?usp=sharing)
- Extract zip in root folder:

```
data/raw/
```

---

##  Execution Pipeline

Run the following scripts in order:

```bash
python scripts/load_data.py
python scripts/build_graph.py
python scripts/train_model.py
```

---

##  Web Application

Launch the Streamlit-based web interface:

```bash
streamlit run webapp/app.py
```

---


