import torch
import numpy as np
from sklearn.datasets import make_moons
from torchvision import datasets
from torchvision import transforms

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

def cifar10_data(device="cpu", seed=0, subset_size=1_000, grayscale=False):
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

    # Create a subset of the first 100 data points
    train_input_subset = torch.utils.data.Subset(train_dataset.data, range(subset_size))
    test_input_subset = torch.utils.data.Subset(test_dataset.data, range(subset_size))
    train_label_subset = torch.utils.data.Subset(train_dataset.targets, range(subset_size))
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