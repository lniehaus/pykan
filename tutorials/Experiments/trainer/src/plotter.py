import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_train_data(dataset, title):
    X = dataset['train_input']
    y = dataset['train_label']
    #plt.scatter(X[:, 0].cpu().detach().numpy(), X[:, 1].cpu().detach().numpy(), c=y[:, 0].cpu().detach().numpy())
    plt.scatter(X[:,0].cpu().detach().numpy(), X[:,1].cpu().detach().numpy(), c=y[:].cpu().detach().numpy())
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    #mlflow.log_figure(plt.gcf(), "train_data.png")
    return plt.gcf()


# def plot_train_data(dataset, title):
#     X = dataset['train_input']
#     y = dataset['train_label']
    
#     # Convert one-hot encoded labels to class indices
#     y_class_indices = torch.argmax(y, dim=1)

#     # Plot the data
#     plt.scatter(X[:, 0].cpu().detach().numpy(), X[:, 1].cpu().detach().numpy(), 
#                 c=y_class_indices.cpu().detach().numpy(), cmap='viridis', edgecolor='k')
    
#     plt.title(title)
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.tight_layout()
#     #mlflow.log_figure(plt.gcf(), "train_data.png")
#     return plt.gcf()


# def plot_train_data(dataset, title):
#     X = dataset['train_input']
#     y = dataset['train_label']
    
#     # Check the shape of y
#     print("Shape of y:", y.shape)

#     # Convert one-hot encoded labels to class indices if y is 2D
#     if y.dim() == 2 and y.size(1) > 1:
#         y_class_indices = torch.argmax(y, dim=1)
#     else:
#         # If y is already 1D or has only one column, use it directly
#         y_class_indices = y.squeeze()  # This will convert [N, 1] to [N] if necessary

#     # Plot the data
#     plt.scatter(X[:, 0].cpu().detach().numpy(), X[:, 1].cpu().detach().numpy(), 
#                 c=y_class_indices.cpu().detach().numpy(), cmap='viridis', edgecolor='k')
    
#     plt.title(title)
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.tight_layout()
#     #mlflow.log_figure(plt.gcf(), "train_data.png")
#     return plt.gcf()

def plot_predictions(model, dataset, title):
    print("Plot Predictions")
    # Make predictions on the test input
    with torch.no_grad():  # Disable gradient calculation for inference
        predictions = model(dataset['test_input'])  # Get model predictions
        predicted_labels = torch.argmax(predictions, dim=1).cpu().detach().numpy()  # Use argmax for class labels
        #print(predicted_labels)

    # Create a scatter plot of the test input colored by the predicted labels
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset['test_input'][:, 0].cpu().detach().numpy(), 
                dataset['test_input'][:, 1].cpu().detach().numpy(), 
                c=predicted_labels, cmap='coolwarm', edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Predicted Label')
    plt.tight_layout()
    #mlflow.log_figure(plt.gcf(), "test_input_predictions.png")
    return plt.gcf()

def plot_mean_std(model, title):
    act_means = []
    act_stds = []
    act_summed_means = []
    act_summed_stds = []
    coef_means = []
    coef_stds = []

    for layer_index, (act_fun, preacts, postacts, postsplines) in enumerate(zip(model.act_fun, model.spline_preacts, model.spline_postacts, model.spline_postsplines)):
        coef = act_fun.coef
        with torch.no_grad():
            act_means.append(postacts.mean().cpu().numpy())
            act_stds.append(postacts.std().cpu().numpy())

            act_summed = postacts.sum(axis=2, keepdims=True)
            act_summed_means.append(act_summed.mean().cpu().numpy())
            act_summed_stds.append(act_summed.std().cpu().numpy())

            coef_means.append(coef.mean().cpu().numpy())
            coef_stds.append(coef.std().cpu().numpy())

    fig, axs = plt.subplots(3,2,figsize=(12,6))
    axs[0][0].plot(act_means, label=f"Act Means: {np.mean(act_means):2e}")
    axs[0][0].set_title(title)
    axs[0][0].set_xlabel("Layer Index")
    axs[0][0].set_ylabel("Mean Value")
    axs[0][0].legend()
    axs[0][1].plot(act_stds, label=f"Act Stds: {np.mean(act_stds):2e}")
    axs[0][1].set_title(title)
    axs[0][1].set_xlabel("Layer Index")
    axs[0][1].set_ylabel("Standard Deviation")
    axs[0][1].legend()

    axs[1][0].plot(act_means, label=f"Act Summed means: {np.mean(act_summed_means):2e}")
    axs[1][0].set_title(title)
    axs[1][0].set_xlabel("Layer Index")
    axs[1][0].set_ylabel("Mean Value")
    axs[1][0].legend()
    axs[1][1].plot(act_stds, label=f"Act Summed Stds: {np.mean(act_summed_means):2e}")
    axs[1][1].set_title(title)
    axs[1][1].set_xlabel("Layer Index")
    axs[1][1].set_ylabel("Standard Deviation")
    axs[1][1].legend()

    axs[2][0].plot(coef_means, label=f"Coef Means: {np.mean(coef_means):2e}")
    axs[2][0].set_title(title)
    axs[2][0].set_xlabel("Layer Index")
    axs[2][0].set_ylabel("Mean Value")
    axs[2][0].legend()
    axs[2][1].plot(coef_stds, label=f"Coef Stds: {np.mean(coef_stds):2e}")
    axs[2][1].set_title(title)
    axs[2][1].set_xlabel("Layer Index")
    axs[2][1].set_ylabel("Standard Deviation")
    axs[2][1].legend()

    plt.tight_layout()
    return fig

def plot_violins(model, title, mode="act", sample_size=100):
    data = []
    for layer_index, (act_fun, acts, preacts, postacts, postsplines) in enumerate(zip(model.act_fun, model.acts, model.spline_preacts, model.spline_postacts, model.spline_postsplines)):
        
        coef = act_fun.coef

        dist_np = None  # Changed from acts_np to dist_np
        if mode == "coef":
            dist_np = coef.cpu().detach().numpy()
        elif mode == "act":
            dist_np = postacts.cpu().detach().numpy()
        elif mode == "grad":
            dist_np = coef.grad.cpu().detach().numpy()

        # Ensure dist_np is a 1D array
        dist_np = dist_np.flatten()

        # Sample a subset of activations if there are more than sample_size
        if len(dist_np) > sample_size:
            sampled_acts = np.random.choice(dist_np, sample_size, replace=False)
        else:
            sampled_acts = dist_np  # Use all if less than sample_size
        
        # Append layer index and sampled activations to the data list
        data.extend([(layer_index, act) for act in sampled_acts])

    # Convert the data into a DataFrame
    df = pd.DataFrame(data, columns=['Layer', 'Activation'])

    plt.figure(figsize=(12, 6))

    # Create a violin plot
    sns.violinplot(data=df, x="Layer", y="Activation", inner="quart")

    # Adding labels and title
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel(f'{mode}')
    plt.tight_layout()

    # Show plot
    plt.show()
    return plt.gcf()

def plot_violins_extended(model, dataset, title, sample_size=100):
    # Assuming model.act_fun and model.spline_preacts are defined
    data = []

    # Collect activations and labels
    for layer_index, (act_fun, acts, preacts, postacts, postsplines) in enumerate(zip(model.act_fun, model.acts, model.spline_preacts, model.spline_postacts, model.spline_postsplines)):
        acts_np = torch.flatten(postacts).cpu().numpy()
        
        # Sample a subset of activations if there are more than sample_size
        if len(acts_np) > sample_size:
            sampled_acts = np.random.choice(acts_np, sample_size, replace=False)
        else:
            sampled_acts = acts_np  # Use all if less than sample_size
        
        # Get corresponding labels for the sampled activations
        labels = dataset['test_label'].cpu().numpy().flatten()
        
        # Sample corresponding labels
        if len(labels) > sample_size:
            sampled_labels = np.random.choice(labels, sample_size, replace=False)
        else:
            sampled_labels = labels  # Use all if less than sample_size
        
        # Append layer index, sampled activations, and labels to the data list
        data.extend([(layer_index, act, label) for act, label in zip(sampled_acts, sampled_labels)])

    # Convert the data into a DataFrame
    df = pd.DataFrame(data, columns=['Layer', 'Activation', 'Label'])

    # Define a color mapping
    color_mapping = {0: 'red', 1: 'orange'}  # Assuming labels are 0 and 1

    # Create a new column for colors based on the mapping
    df['Color'] = df['Label'].map(color_mapping)

    # Create a violin plot
    g = sns.catplot(data=df, x="Layer", y="Activation", kind="violin", inner=None, height=6, aspect=1.5)

    # Overlay a swarm plot with custom colors
    sns.swarmplot(data=df, x="Layer", y="Activation", hue="Label", palette=color_mapping, size=3, ax=g.ax)

    # Adding labels and title
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Activations')
    plt.tight_layout()

    # Show plot
    #plt.show()
    return plt.gcf()

def plot_summed_violins(model, title, mode="act", sample_size=100):
    data = []
    grid_ranges = []  # To store grid ranges for each layer

    for layer_index, (act_fun, acts, preacts, postacts, postsplines) in enumerate(zip(model.act_fun, model.acts, model.spline_preacts, model.spline_postacts, model.spline_postsplines)):
        
        coef = act_fun.coef

        dist_np = None  # Changed from acts_np to dist_np
        if mode == "act":
            dist_np = coef.cpu().detach().numpy()
            dist_np = dist_np.sum(axis=2, keepdims=True)

        # Ensure dist_np is a 1D array
        dist_np = dist_np.flatten()

        # Sample a subset of activations if there are more than sample_size
        if len(dist_np) > sample_size:
            sampled_acts = np.random.choice(dist_np, sample_size, replace=False)
        else:
            sampled_acts = dist_np  # Use all if less than sample_size
        
        # Append layer index and sampled activations to the data list
        data.extend([(layer_index, act) for act in sampled_acts])

        # Store the grid range for this layer
        grid_range = act_fun.grid_range
        grid_range_extended = act_fun.grid_range_extended
        grid_ranges.append((grid_range,grid_range_extended))
        

    # Convert the data into a DataFrame
    df = pd.DataFrame(data, columns=['Layer', 'Activation'])

    plt.figure(figsize=(12, 6))

    # Create a violin plot
    sns.violinplot(data=df, x="Layer", y="Activation", inner="quart")

    # Adding horizontal lines for each layer's grid range
    for layer_index, (grid_range, grid_range_extended) in enumerate(grid_ranges):
        dist = 1/len(grid_ranges)
        xmin = layer_index * dist
        xmax = (layer_index * dist) + dist
        plt.axhline(y=grid_range[0], xmin=xmin, xmax=xmax, color='blue', linestyle='--', label=f'Grid Range, Layer {layer_index} Lower Bound' if layer_index == 0 else "")
        plt.axhline(y=grid_range[1], xmin=xmin, xmax=xmax, color='blue', linestyle='--', label=f'Grid Range, Layer {layer_index} Lower Bound' if layer_index == 0 else "")
        plt.axhline(y=grid_range_extended[0], xmin=xmin, xmax=xmax, color='red', linestyle='--', label=f'Grid Range Extended, Layer {layer_index} Upper Bound' if layer_index == 0 else "")
        plt.axhline(y=grid_range_extended[1], xmin=xmin, xmax=xmax, color='red', linestyle='--', label=f'Grid Range Extended, Layer {layer_index} Upper Bound' if layer_index == 0 else "")


    # Adding labels and title
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel(f'{mode}')
    plt.tight_layout()

    # Show plot
    #plt.show()
    return plt.gcf()

def plot_layerwise_postacts_and_postsplines(model, title, sample_size=100):
    """
    Plots a violin plot for each layer, showing both spline_postacts and spline_postsplines as separate violins.
    """
    data = []
    for layer_index, (postacts, postsplines) in enumerate(zip(model.spline_postacts, model.spline_postsplines)):
        # Flatten and sample postacts
        postacts_np = postacts.cpu().detach().numpy().flatten()
        if len(postacts_np) > sample_size:
            sampled_postacts = np.random.choice(postacts_np, sample_size, replace=False)
        else:
            sampled_postacts = postacts_np
        data.extend([(layer_index, act, 'postacts') for act in sampled_postacts])

        # Flatten and sample postsplines
        postsplines_np = postsplines.cpu().detach().numpy().flatten()
        if len(postsplines_np) > sample_size:
            sampled_postsplines = np.random.choice(postsplines_np, sample_size, replace=False)
        else:
            sampled_postsplines = postsplines_np
        data.extend([(layer_index, act, 'postsplines') for act in sampled_postsplines])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Layer', 'Activation', 'Type'])

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x="Layer", y="Activation", hue="Type", split=True, inner="quart", palette={"postacts": "skyblue", "postsplines": "orange"})
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Activation')
    plt.tight_layout()
    return plt.gcf()
