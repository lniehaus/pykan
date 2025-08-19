import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = []
        self.hooks = []
        
        # Register hooks for each layer
        for layer in model.children():
            hook = layer.register_forward_hook(self.get_features)
            self.hooks.append(hook)

    def get_features(self, module, input, output):
        self.features.append(output)

    def clear(self):
        self.features = []

    def close(self):
        for hook in self.hooks:
            hook.remove()

def train_linear_classifier(features, labels, num_classes, num_epochs=100, learning_rate=0.01):
    input_size = features.shape[1]
    classifier = nn.Linear(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return classifier

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 784  # Example for MNIST
    hidden_sizes = [128, 64]
    output_size = 10  # Number of classes
    num_classes = 10
    num_epochs = 100
    learning_rate = 0.01

    # Create a dataset (dummy data for illustration)
    X = torch.randn(1000, input_size)  # 1000 samples
    y = torch.randint(0, num_classes, (1000,))  # Random labels

    # Initialize model and feature extractor
    model = MLP(input_size, hidden_sizes, output_size)
    feature_extractor = FeatureExtractor(model)

    # Forward pass to extract features
    model(X)
    features = feature_extractor.features

    # Train a linear classifier for each layer's output
    for i, feature in enumerate(features):
        print(f"Training linear classifier for layer {i + 1}")
        classifier = train_linear_classifier(feature.detach(), y, num_classes, num_epochs, learning_rate)

    # Clean up
    feature_extractor.close()