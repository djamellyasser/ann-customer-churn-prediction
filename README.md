# ANN Customer Churn Prediction

A Python project for predicting customer churn using Artificial Neural Networks (ANN). This project refactors a Jupyter Notebook into a structured, production-ready codebase.

## Project Structure

```
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Processed data
├── notebooks/                # Original analysis notebook
├── src/                      # Source code
│   ├── data/                 # Data loading scripts
│   ├── features/             # Feature engineering scripts
│   ├── models/               # ANN model definition and training
│   └── visualization/        # Plots and graphs
├── main.py                   # Main pipeline entry point
└── requirements.txt          # Project dependencies
```

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/djamellyasser/ann-customer-churn-prediction.git
   cd ann-customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the full pipeline (Data Loading -> Processing -> Training -> Evaluation -> Prediction):

```bash
python main.py
```

## Model Details

- **Architecture**: 3-layer ANN (Dense(6, relu) -> Dense(6, relu) -> Dense(1, sigmoid))
- **Input**: Customer demographics, credit score, balance, etc.
- **Output**: Binary classification (0: Stay, 1: Exit)
