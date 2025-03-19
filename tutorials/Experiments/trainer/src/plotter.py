import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_train_data(dataset, title="Train Datase"):
    X = dataset['train_input']
    y = dataset['train_label']
    plt.scatter(X[:, 0].cpu().detach().numpy(), X[:, 1].cpu().detach().numpy(), c=y[:, 0].cpu().detach().numpy())
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    #mlflow.log_figure(plt.gcf(), "train_data.png")
    return plt.gcf()

def plot_predictions(model, dataset):
    print("Plot Predictions")
    # Make predictions on the test input
    with torch.no_grad():  # Disable gradient calculation for inference
        predictions = model(dataset['test_input'])  # Get model predictions
        predicted_labels = torch.round(predictions[:, 0]).cpu().detach().numpy()  # Round predictions to get class labels

    # Create a scatter plot of the test input colored by the predicted labels
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset['test_input'][:, 0].cpu().detach().numpy(), 
                dataset['test_input'][:, 1].cpu().detach().numpy(), 
                c=predicted_labels, cmap='coolwarm', edgecolor='k', s=20)
    plt.title('Test Input Colored by Model Predictions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Predicted Label')
    #mlflow.log_figure(plt.gcf(), "test_input_predictions.png")
    return plt.gcf()
