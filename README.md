# MLOps Lab 5: W&B Experiment Tracking
##  📌 Overview
This project demonstrates the implementation of experiment tracking and artifact management for an MNIST digit classifier using Weights & Biases (W&B).

## 📂 Project Structure
- `train.py`: Main script for model training and W&B experiment tracking.
- `config.yaml`: Configuration file containing hyperparameters and data settings.
- `requirements.txt`: List of required Python libraries (wandb, scikit-learn, etc.).
- `.gitignore`: Configuration to exclude unnecessary files (.idea/, venv/, *.pkl) from the repository
## 🚀 How to Run
### 1. Install requirements:
 ```bash
pip install -r requirements.txt
```

### 2. Set Credentials:

Set `WANDB_API_KEY` in your system or IDE environment variables for automatic authentication. 
### 3. Execute Training:
```bash
python train.py
```
To run different experiments, modify the values in `config.yaml` before starting the script
