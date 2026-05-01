import wandb
import yaml
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def train():

    # Load configuration from YAML
    try:
        with open("config.yaml", "r") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure the file exists.")
        return

    # Structured run naming for easier comparison in W&B UI
    run_name = f"RF-trees_{config_data['model']['n_estimators']}-depth_{config_data['model']['max_depth']}"

    # Initialize W&B run
    # Credentials (WANDB_API_KEY) are automatically pulled from PyCharm Environment Variables
    run = wandb.init(
        project=config_data['project_name'],
        config=config_data,
        name=run_name
    )

    # Load and Split MNIST dataset
    print("Fetching MNIST dataset from OpenML...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    # Use split parameters defined in config.yaml
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config_data['data']['test_size'],
        random_state=config_data['model']['random_state']
    )

    # Train Random Forest Classifier
    print(f"Training model: {run_name}...")
    model = RandomForestClassifier(
        n_estimators=config_data['model']['n_estimators'],
        max_depth=config_data['model']['max_depth'],
        random_state=config_data['model']['random_state']
    )
    model.fit(X_train, y_train)

    # Evaluate and Log Metrics
    y_pred = model.predict(X_test)

    # Type conversion to ensure compatibility with W&B visualization tools
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)

    acc = accuracy_score(y_test, y_pred)
    run.log({"accuracy": acc})
    print(f"Run Accuracy: {acc:.4f}")

    # Log Visual Chart: Confusion Matrix
    run.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test,
        preds=y_pred,
        class_names=[str(i) for i in range(10)]
    )})

    # Log Model Artifact
    # Save the model locally before uploading
    model_path = "mnist_rf_model.pkl"
    joblib.dump(model, model_path)

    # Register the model file as a versioned artifact in W&B
    artifact = wandb.Artifact(
        name='mnist-classifier-model',
        type='model',
        description='Random Forest model for MNIST digits classification'
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    print("Model artifact successfully uploaded to W&B.")

    # Finish W&B session
    run.finish()


if __name__ == "__main__":
    train()