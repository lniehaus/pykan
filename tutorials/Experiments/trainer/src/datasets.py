import torch
import numpy as np
from sklearn.datasets import make_moons
from torchvision import datasets
from torchvision import transforms
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mnist1d.data import make_dataset, get_dataset_args

def make_classification_data(n_samples=1000, n_features=2, n_labels=1, n_informative=2, n_redundant=10, seed=0, device="cpu"):
    np.random.seed(seed)

    # Generate a synthetic dataset
    X, y = make_classification(n_samples=n_samples,         # Number of samples
                            n_features=n_features,          # Total number of features
                            n_informative=n_informative,    # Number of informative features
                            n_redundant=n_redundant,        # Number of redundant features
                            n_classes=n_labels,             # Number of classes
                            n_clusters_per_class=1,
                            random_state=seed)

    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    train_input, test_input, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=seed)

    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)

    return dataset

# def moon_data(data_noise_level, n_samples=1000,seed=0, device="cpu"):
#     # Generate the original dataset
#     train_input, train_label = make_moons(n_samples=n_samples, shuffle=True, noise=data_noise_level, random_state=seed)
#     test_input, test_label = make_moons(n_samples=n_samples, shuffle=True, noise=data_noise_level, random_state=seed+1)



#     # Convert to PyTorch tensors
#     dtype = torch.get_default_dtype()
#     #device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     dataset = {}
#     # dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
#     # dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
#     # dataset['train_label'] = torch.from_numpy(train_label[:, None]).type(dtype).to(device)
#     # dataset['test_label'] = torch.from_numpy(test_label[:, None]).type(dtype).to(device)

#     # dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
#     # dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
#     # dataset['train_label'] = torch.from_numpy(train_label).type(torch.long).to(device)
#     # dataset['test_label'] = torch.from_numpy(test_label).type(torch.long).to(device)

#     dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
#     dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
#     dataset['train_label'] = torch.from_numpy(train_label).type(dtype).to(device)
#     dataset['test_label'] = torch.from_numpy(test_label).type(dtype).to(device)

#     return dataset


def moon_data(data_noise_level, n_samples=1000, seed=0, device="cpu"):

    train_input, train_label = make_moons(n_samples=n_samples, shuffle=True, noise=data_noise_level, random_state=seed)
    test_input, test_label = make_moons(n_samples=n_samples, shuffle=False, noise=data_noise_level, random_state=seed+1)

    # Min-max scale both train and test to [-1, 1]
    all_data = np.vstack([train_input, test_input])
    min_vals = all_data.min(axis=0)
    max_vals = all_data.max(axis=0)
    scale = 2 / (max_vals - min_vals)
    shift = -1 - min_vals * scale

    train_input = train_input * scale + shift
    test_input = test_input * scale + shift

    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_label).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_label).type(torch.long).to(device)

    return dataset

# def moon_data(data_noise_level, n_samples=1000, seed=0, device="cpu"):
#     # Generate the original dataset
#     train_input, train_label = make_moons(n_samples=n_samples, shuffle=True, noise=data_noise_level, random_state=seed)
#     test_input, test_label = make_moons(n_samples=n_samples, shuffle=False, noise=data_noise_level, random_state=seed+1)

#     # Convert to PyTorch tensors
#     dtype = torch.get_default_dtype()
#     dataset = {}
#     dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
#     dataset['train_label'] = torch.from_numpy(train_label).type(dtype).to(device).unsqueeze(1)
#     dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
#     dataset['test_label'] = torch.from_numpy(test_label).type(dtype).to(device).unsqueeze(1)

#     # # One-hot encode the labels
#     # num_classes = 2  # For the moons dataset, there are 2 classes
#     # dataset['train_label'] = torch.nn.functional.one_hot(torch.from_numpy(train_label).type(torch.int64), num_classes=num_classes).to(device)
#     # dataset['test_label'] = torch.nn.functional.one_hot(torch.from_numpy(test_label).type(torch.int64), num_classes=num_classes).to(device)

#     # dataset['train_input'] = dataset['train_input'].type(dtype)
#     # dataset['test_input'] = dataset['test_input'].type(dtype)
#     # dataset['train_label'] = dataset['train_label'].type(dtype)
#     # dataset['test_label'] = dataset['test_label'].type(dtype)


#     # print(dataset['train_input'].type())
#     # print(dataset['test_input'].type())
#     # print(dataset['train_label'].type())
#     # print(dataset['test_label'].type())

#     # # dtype = torch.get_default_dtype()
#     # # dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
#     # # dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
#     # # dataset['train_label'] = torch.from_numpy(train_label).type(torch.long).to(device)
#     # # dataset['test_label'] = torch.from_numpy(test_label).type(torch.long).to(device)

#     # print("moon dataset['train_input'][0:10]", dataset['train_input'].shape, dataset['train_input'][0:100])
#     # print("moon dataset['train_label'][0:10]", dataset['train_label'].shape, dataset['train_label'][0:100])

#     return dataset

# def random_data(distribution, n_samples=1000, n_features=2, n_labels=1, loc=0.0, normal_scale=1.0, range=(-1,1), seed=0, device="cpu"):
#     np.random.seed(seed)
#     # Generate random data based on the specified distribution
#     if distribution == 'uniform':
#         # Generate random data from a uniform distribution
#         train_input = np.random.uniform(low=range[0], high=range[1], size=(n_samples, n_features))
#         test_input = np.random.uniform(low=range[0], high=range[1], size=(n_samples, n_features))
#     elif distribution == 'normal':
#         # Generate random data from a normal distribution
#         train_input = np.random.normal(loc=loc, scale=normal_scale, size=(n_samples, n_features))
#         test_input = np.random.normal(loc=loc, scale=normal_scale, size=(n_samples, n_features))
#     else:
#         raise ValueError("Invalid data_distribution. Choose 'uniform' or 'normal'.")

#         # Generate random labels (0 or 1) for the dataset

#     train_label = np.random.randint(0, 2, size=(n_samples, n_labels))  # Random labels for training
#     test_label = np.random.randint(0, 2, size=(n_samples, n_labels))   # Random labels for testing

#     # Convert to PyTorch tensors
#     dtype = torch.get_default_dtype()
#     dataset = {}
#     dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
#     dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
#     dataset['train_label'] = torch.from_numpy(train_label).type(dtype).to(device)
#     dataset['test_label'] = torch.from_numpy(test_label).type(dtype).to(device)

#     print("dataset['test_label'][0]", dataset['test_label'][0])
#     print("dataset['test_label'][1]", dataset['test_label'][1])
#     print("dataset['test_label'][2]", dataset['test_label'][2])
#     print("dataset['test_label'][3]", dataset['test_label'][3])
#     print("dataset['test_label'][4]", dataset['test_label'][4])
#     print("dataset['test_label'][5]", dataset['test_label'][5])

#     return dataset

def random_data(distribution, n_samples=1000, n_features=2, n_labels=1, loc=0.0, normal_scale=1.0, range=(-1, 1), seed=0, device="cpu"):
    np.random.seed(seed)

    # Generate random input data based on the specified distribution
    if distribution == 'uniform':
        train_input = np.random.uniform(low=range[0], high=range[1], size=(n_samples, n_features))
        test_input = np.random.uniform(low=range[0], high=range[1], size=(n_samples, n_features))
    elif distribution == 'normal':
        train_input = np.random.normal(loc=loc, scale=normal_scale, size=(n_samples, n_features))
        test_input = np.random.normal(loc=loc, scale=normal_scale, size=(n_samples, n_features))
    else:
        raise ValueError("Invalid distribution. Choose 'uniform' or 'normal'.")

    # Generate random output labels as integers in the range [0, n_labels)
    train_labels = np.random.randint(low=0, high=n_labels, size=(n_samples, 1)).squeeze(1)
    test_labels = np.random.randint(low=0, high=n_labels, size=(n_samples, 1)).squeeze(1)

    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)

    return dataset

# def random_data(distribution, n_samples=1000, n_features=2, n_labels=1, loc=0.0, normal_scale=1.0, range=(-1, 1), seed=0, device="cpu"):
#     np.random.seed(seed)

#     # Generate random data based on the specified distribution
#     if distribution == 'uniform':
#         # Generate random data from a uniform distribution
#         train_input = np.random.uniform(low=range[0], high=range[1], size=(n_samples, n_features))
#         test_input = np.random.uniform(low=range[0], high=range[1], size=(n_samples, n_features))
#     elif distribution == 'normal':
#         # Generate random data from a normal distribution
#         train_input = np.random.normal(loc=loc, scale=normal_scale, size=(n_samples, n_features))
#         test_input = np.random.normal(loc=loc, scale=normal_scale, size=(n_samples, n_features))
#     else:
#         raise ValueError("Invalid data_distribution. Choose 'uniform' or 'normal'.")

#     # Generate random labels (one-hot encoded)
#     if n_labels > 1:
#         # Create one-hot encoded labels
#         train_labels = np.zeros((n_samples, n_labels), dtype=int)
#         test_labels = np.zeros((n_samples, n_labels), dtype=int)
        
#         # Randomly select one label to be 1 for each sample
#         train_indices = np.random.choice(n_labels, n_samples)
#         test_indices = np.random.choice(n_labels, n_samples)
        
#         train_labels[np.arange(n_samples), train_indices] = 1
#         test_labels[np.arange(n_samples), test_indices] = 1
#     else:
#         # If n_labels is 1, generate binary labels
#         train_labels = np.random.randint(0, 2, size=(n_samples, n_labels))
#         test_labels = np.random.randint(0, 2, size=(n_samples, n_labels))

#     # Convert to PyTorch tensors
#     dtype = torch.get_default_dtype()
#     dataset = {}
#     dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
#     dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
#     dataset['train_label'] = torch.from_numpy(train_labels).type(dtype).to(device)
#     dataset['test_label'] = torch.from_numpy(test_labels).type(dtype).to(device)

#     print(dataset['train_input'].type())
#     print(dataset['test_input'].type())
#     print(dataset['train_label'].type())
#     print(dataset['test_label'].type())

#     print("random dataset['train_input']", dataset['train_input'].shape, dataset['train_input'][0])
#     print("random dataset['train_label']", dataset['train_label'].shape, dataset['train_label'][0])

#     return dataset

def mnist_data(device="cpu", seed=0):
    #https://github.com/ale93111/pykan_mnist/blob/main/kan_mnist.ipynb

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Define data transformation to normalize pixel values between 0 and 1
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    # Load MNIST dataset using torchvision
    train_dataset = datasets.MNIST('./datasets/MNIST_DATA', download=True, train=True, transform=transform)
    test_dataset = datasets.MNIST('./datasets/MNIST_DATA', download=True, train=False, transform=transform)

    # Convert to PyTorch tensors and move data to the specified device
    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = train_dataset.data.type(dtype).to(device) / 255.0  # Normalize pixel values between 0 and 1
    dataset['test_input'] = test_dataset.data.type(dtype).to(device) / 255.0   # Normalize pixel values between 0 and 1
    dataset['train_label'] = train_dataset.targets.to(device)
    dataset['test_label'] = test_dataset.targets.to(device)

    # Flatten Data
    dataset['train_input'] = dataset['train_input'].flatten(1)
    dataset['test_input'] = dataset['test_input'].flatten(1)

    return dataset

def cifar10_data(device="cpu", seed=0, subset_size=100_000, grayscale=False):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    transform = None
    if grayscale:
    # Define data transformation to normalize pixel values between 0 and 1
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB channels
        ]) 
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB channels
        ]) 

    # Load CIFAR-10 dataset using torchvision
    train_dataset = datasets.CIFAR10('./datasets/CIFAR10_DATA', download=True, train=True, transform=transform)
    test_dataset = datasets.CIFAR10('./datasets/CIFAR10_DATA', download=True, train=False, transform=transform)

    train_input_subset = train_dataset.data
    test_input_subset = test_dataset.data
    train_label_subset = train_dataset.targets
    test_label_subset = test_dataset.targets

    if subset_size <= len(train_dataset):
        # Create a subset of the first subset_size data points
        train_input_subset = torch.utils.data.Subset(train_dataset.data, range(subset_size))
        train_label_subset = torch.utils.data.Subset(train_dataset.targets, range(subset_size))
        test_input_subset = torch.utils.data.Subset(test_dataset.data, range(subset_size))
        test_label_subset = torch.utils.data.Subset(test_dataset.targets, range(subset_size))


    # Convert to PyTorch tensors and move data to the specified device
    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.tensor(train_input_subset, dtype=torch.float32).to(device) / 255.0  # Normalize pixel values between 0 and 1
    dataset['test_input'] = torch.tensor(test_input_subset, dtype=torch.float32).to(device) / 255.0    # Normalize pixel values between 0 and 1
    dataset['train_label'] = torch.tensor(train_label_subset).to(device)
    dataset['test_label'] = torch.tensor(test_label_subset).to(device)

    # # Flatten Data
    # dataset['train_input'] = dataset['train_input'].permute(0, 3, 1, 2).flatten(1)  # Change to (N, C, H, W) and flatten
    # dataset['test_input'] = dataset['test_input'].permute(0, 3, 1, 2).flatten(1)    # Change to (N, C, H, W) and flatten

    dataset['train_input'] = dataset['train_input'].flatten(1)  # Change to (N, C, H, W) and flatten
    dataset['test_input'] = dataset['test_input'].flatten(1)    # Change to (N, C, H, W) and flatten

    # print("dataset['train_input'] ", dataset['train_input'].shape)
    # print("dataset['test_input'] ", dataset['test_input'].shape)
    # print("dataset['train_label'] ", dataset['train_label'].shape)
    # print("dataset['test_label'] ", dataset['test_label'].shape)


    return dataset


# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import Subset

# def cifar10_data(device="cpu", seed=0, num_samples=100):
#     # Set the random seed for reproducibility
#     torch.manual_seed(seed)

#     # Define data transformation to normalize pixel values between 0 and 1
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB channels
#     ])
    
#     # Load CIFAR-10 dataset using torchvision
#     train_dataset = datasets.CIFAR10('./datasets/CIFAR10_DATA', download=True, train=True, transform=transform)
#     test_dataset = datasets.CIFAR10('./datasets/CIFAR10_DATA', download=True, train=False, transform=transform)

#     # If you want to limit the number of samples, use Subset
#     if num_samples < len(train_dataset):
#         indices = torch.randperm(len(train_dataset))[:num_samples]
#         train_dataset = Subset(train_dataset, indices)

#     # Convert to PyTorch tensors and move data to the specified device
#     dataset = {}
#     dataset['train_input'] = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).to(device)
#     dataset['test_input'] = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
#     dataset['train_label'] = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))]).to(device)
#     dataset['test_label'] = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))]).to(device)

#     return dataset


def mnist1d_data(device="cpu", seed=0, subset_size=1_000_000):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    dtype = torch.get_default_dtype()

    defaults = get_dataset_args()
    data = make_dataset(defaults)

    train_data = data['x'][:subset_size]
    test_data = data['x_test'][:subset_size]
    train_labels = data['y'][:subset_size]
    test_labels = data['y_test'][:subset_size]

    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_data).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_data).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)

    return dataset


# def boxes_2d_dataset(n_classes=16, datapoints_per_class=10, bounds=(-1, 1, -1, 1), device="cpu", seed=42):
#     dtype = torch.get_default_dtype()
#     np.random.seed(seed)
#     # Compute grid size (try to make it as square as possible)
#     grid_size = int(np.ceil(np.sqrt(n_classes)))
#     x_edges = np.linspace(bounds[0], bounds[1], grid_size + 1)
#     y_edges = np.linspace(bounds[2], bounds[3], grid_size + 1)
#     data = []
#     labels = []
#     class_idx = 0
#     for i in range(grid_size):
#         for j in range(grid_size):
#             if class_idx >= n_classes:
#                 break
#             x_min, x_max = x_edges[i], x_edges[i+1]
#             y_min, y_max = y_edges[j], y_edges[j+1]
#             # Sample datapoints_per_class points uniformly within this cell
#             xs = np.random.uniform(x_min, x_max, size=(datapoints_per_class, 1))
#             ys = np.random.uniform(y_min, y_max, size=(datapoints_per_class, 1))
#             points = np.hstack([xs, ys])
#             data.append(points)
#             labels.extend([class_idx] * datapoints_per_class)
#             class_idx += 1
#         if class_idx >= n_classes:
#             break
#     data = np.vstack(data)
#     labels = np.array(labels)

#     train_input, test_input, train_label, test_label = train_test_split(
#         data, labels, test_size=0.2, random_state=42, stratify=labels
#     )

#     dataset = {}
#     dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
#     dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
#     dataset['train_label'] = torch.from_numpy(train_label).type(torch.long).to(device)
#     dataset['test_label'] = torch.from_numpy(test_label).type(torch.long).to(device)
#     return dataset

def boxes_2d_dataset(n_classes=16, datapoints_per_class=10, bounds=(-1, 1, -1, 1), device="cpu", seed=42):
    dtype = torch.get_default_dtype()
    np.random.seed(seed)
    # Compute grid size (try to make it as square as possible)
    grid_size = int(np.ceil(np.sqrt(n_classes)))
    x_edges = np.linspace(bounds[0], bounds[1], grid_size + 1)
    y_edges = np.linspace(bounds[2], bounds[3], grid_size + 1)
    # Shuffle class indices to randomize which class is at which patch
    class_indices = np.arange(n_classes)
    np.random.shuffle(class_indices)
    data = []
    labels = []
    class_idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if class_idx >= n_classes:
                break
            x_min, x_max = x_edges[i], x_edges[i+1]
            y_min, y_max = y_edges[j], y_edges[j+1]
            # Sample datapoints_per_class points uniformly within this cell
            xs = np.random.uniform(x_min, x_max, size=(datapoints_per_class, 1))
            ys = np.random.uniform(y_min, y_max, size=(datapoints_per_class, 1))
            points = np.hstack([xs, ys])
            data.append(points)
            # Assign the shuffled class index
            labels.extend([class_indices[class_idx]] * datapoints_per_class)
            class_idx += 1
        if class_idx >= n_classes:
            break
    data = np.vstack(data)
    labels = np.array(labels)

    train_input, test_input, train_label, test_label = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_label).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_label).type(torch.long).to(device)
    return dataset

def boxes_2d_dataset(
    n_classes=16,
    datapoints_per_class=10,
    bounds=(-1, 1, -1, 1),
    device="cpu",
    seed=42,
    distribution="uniform",  # "uniform" or "normal"
    normal_mean=0.0,
    normal_std=None  # If None, will be calculated to avoid overlap
):
    dtype = torch.get_default_dtype()
    np.random.seed(seed)
    grid_size = int(np.ceil(np.sqrt(n_classes)))
    x_edges = np.linspace(bounds[0], bounds[1], grid_size + 1)
    y_edges = np.linspace(bounds[2], bounds[3], grid_size + 1)
    class_indices = np.arange(n_classes)
    np.random.shuffle(class_indices)
    data = []
    labels = []
    class_idx = 0

    # Calculate normal_std if needed to avoid overlap
    if distribution == "normal" and normal_std is None:
        # Distance between centers
        x_cell = (bounds[1] - bounds[0]) / grid_size
        y_cell = (bounds[3] - bounds[2]) / grid_size
        # Use 3 std devs to cover 99.7% of data, so 3*std < 0.5*cell_size
        # (so 99.7% of points are within half the cell)
        normal_std = min(x_cell, y_cell) / 6.0

    for i in range(grid_size):
        for j in range(grid_size):
            if class_idx >= n_classes:
                break
            x_min, x_max = x_edges[i], x_edges[i+1]
            y_min, y_max = y_edges[j], y_edges[j+1]
            if distribution == "uniform":
                xs = np.random.uniform(x_min, x_max, size=(datapoints_per_class, 1))
                ys = np.random.uniform(y_min, y_max, size=(datapoints_per_class, 1))
            elif distribution == "normal":
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                xs = np.random.normal(loc=x_center + normal_mean, scale=normal_std, size=(datapoints_per_class, 1))
                ys = np.random.normal(loc=y_center + normal_mean, scale=normal_std, size=(datapoints_per_class, 1))
                xs = np.clip(xs, x_min, x_max)
                ys = np.clip(ys, y_min, y_max)
            else:
                raise ValueError("distribution must be 'uniform' or 'normal'")
            points = np.hstack([xs, ys])
            data.append(points)
            labels.extend([class_indices[class_idx]] * datapoints_per_class)
            class_idx += 1
        if class_idx >= n_classes:
            break
    data = np.vstack(data)
    labels = np.array(labels)

    train_input, test_input, train_label, test_label = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_label).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_label).type(torch.long).to(device)
    return dataset


def and_data(n_samples=1000, noise=0.0, seed=0, device="cpu"):
    np.random.seed(seed)
    # Generate all possible combinations for AND, OR, XOR
    X = np.random.randint(0, 2, size=(n_samples * 2, 2))
    y = np.logical_and(X[:, 0], X[:, 1]).astype(int)
    if noise > 0:
        flip = np.random.rand(len(y)) < noise
        y[flip] = 1 - y[flip]
    # Split into train and test sets (half-half, no overlap)
    split = len(X) // 2
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]
    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_X).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_X).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_y).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_y).type(torch.long).to(device)
    return dataset

def or_data(n_samples=1000, noise=0.0, seed=0, device="cpu"):
    np.random.seed(seed)
    X = np.random.randint(0, 2, size=(n_samples * 2, 2))
    y = np.logical_or(X[:, 0], X[:, 1]).astype(int)
    if noise > 0:
        flip = np.random.rand(len(y)) < noise
        y[flip] = 1 - y[flip]
    split = len(X) // 2
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]
    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_X).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_X).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_y).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_y).type(torch.long).to(device)
    return dataset

def xor_data(n_samples=1000, noise=0.0, seed=0, device="cpu"):
    np.random.seed(seed)
    X = np.random.randint(0, 2, size=(n_samples * 2, 2))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    if noise > 0:
        flip = np.random.rand(len(y)) < noise
        y[flip] = 1 - y[flip]
    split = len(X) // 2
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]
    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_X).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_X).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_y).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_y).type(torch.long).to(device)
    return dataset
