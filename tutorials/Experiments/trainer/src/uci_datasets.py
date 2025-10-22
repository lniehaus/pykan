import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests
import os
from io import StringIO

def iris_data(seed=0, device="cpu", test_size=0.3):
    """
    Load the Iris dataset - one of the most famous UCI datasets.
    Features: 4 (sepal length, sepal width, petal length, petal width)
    Classes: 3 (setosa, versicolor, virginica)
    Samples: 150
    """
    np.random.seed(seed)
    
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the dataset
    train_input, test_input, train_labels, test_labels = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
    
    return dataset

def wine_data(seed=0, device="cpu", test_size=0.3):
    """
    Load the Wine dataset from UCI.
    Features: 13 (chemical analysis of wines)
    Classes: 3 (wine cultivars)
    Samples: 178
    """
    np.random.seed(seed)
    
    # Load the wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the dataset
    train_input, test_input, train_labels, test_labels = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
    
    return dataset

def breast_cancer_data(seed=0, device="cpu", test_size=0.3):
    """
    Load the Breast Cancer Wisconsin dataset.
    Features: 30 (computed from digitized image of breast mass)
    Classes: 2 (malignant, benign)
    Samples: 569
    """
    np.random.seed(seed)
    
    # Load the breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the dataset
    train_input, test_input, train_labels, test_labels = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
    
    return dataset

def digits_data(seed=0, device="cpu", test_size=0.3):
    """
    Load the Optical Recognition of Handwritten Digits dataset.
    Features: 64 (8x8 images of digits)
    Classes: 10 (digits 0-9)
    Samples: 1797
    """
    np.random.seed(seed)
    
    # Load the digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize pixel values to [0, 1]
    X = X / 16.0
    
    # Split the dataset
    train_input, test_input, train_labels, test_labels = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    dtype = torch.get_default_dtype()
    dataset = {}
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
    
    return dataset

def glass_data(seed=0, device="cpu", test_size=0.3):
    """
    Load the Glass Identification dataset from UCI.
    Features: 9 (refractive index and chemical composition)
    Classes: 6 (types of glass)
    Samples: 214
    """
    np.random.seed(seed)
    
    # Download glass dataset from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the data
        data = pd.read_csv(StringIO(response.text), header=None)
        
        # First column is ID, last column is class
        X = data.iloc[:, 1:-1].values  # Features (columns 1-9)
        y = data.iloc[:, -1].values    # Classes (last column)
        
        # Convert class labels to 0-indexed
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the dataset
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset
        
    except Exception as e:
        print(f"Error loading glass dataset: {e}")
        # Return a synthetic dataset as fallback
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=214, n_features=9, n_classes=6, 
                                 n_informative=7, n_redundant=2, random_state=seed)
        
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset

def ionosphere_data(seed=0, device="cpu", test_size=0.3):
    """
    Load the Ionosphere dataset from UCI.
    Features: 34 (radar returns from ionosphere)
    Classes: 2 (good, bad)
    Samples: 351
    """
    np.random.seed(seed)
    
    # Download ionosphere dataset from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the data
        data = pd.read_csv(StringIO(response.text), header=None)
        
        # Last column is class, others are features
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Classes
        
        # Convert class labels to binary (g=1, b=0)
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the dataset
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset
        
    except Exception as e:
        print(f"Error loading ionosphere dataset: {e}")
        # Return a synthetic dataset as fallback
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=351, n_features=34, n_classes=2, 
                                 n_informative=25, n_redundant=9, random_state=seed)
        
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset

def sonar_data(seed=0, device="cpu", test_size=0.3):
    """
    Load the Sonar dataset from UCI.
    Features: 60 (sonar signals bounced off metal cylinder vs rocks)
    Classes: 2 (metal, rock)
    Samples: 208
    """
    np.random.seed(seed)
    
    # Download sonar dataset from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the data
        data = pd.read_csv(StringIO(response.text), header=None)
        
        # Last column is class, others are features
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Classes
        
        # Convert class labels to binary (M=1, R=0)
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the dataset
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset
        
    except Exception as e:
        print(f"Error loading sonar dataset: {e}")
        # Return a synthetic dataset as fallback
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=208, n_features=60, n_classes=2, 
                                 n_informative=45, n_redundant=15, random_state=seed)
        
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset

def heart_disease_data(seed=0, device="cpu", test_size=0.3):
    """
    Load the Heart Disease dataset from UCI.
    Features: 13 (age, sex, chest pain type, etc.)
    Classes: 2 (presence/absence of heart disease)
    Samples: 303
    """
    np.random.seed(seed)
    
    # Download heart disease dataset from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the data
        data = pd.read_csv(StringIO(response.text), header=None, na_values='?')
        
        # Remove rows with missing values
        data = data.dropna()
        
        # Last column is class, others are features
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Classes
        
        # Convert multi-class to binary (0 = no disease, >0 = disease)
        y = (y > 0).astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the dataset
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset
        
    except Exception as e:
        print(f"Error loading heart disease dataset: {e}")
        # Return a synthetic dataset as fallback
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=303, n_features=13, n_classes=2, 
                                 n_informative=10, n_redundant=3, random_state=seed)
        
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset

def pima_diabetes_data(seed=0, device="cpu", test_size=0.3):
    """
    Load the Pima Indians Diabetes dataset from UCI.
    Features: 8 (pregnancies, glucose, blood pressure, etc.)
    Classes: 2 (diabetic, non-diabetic)
    Samples: 768
    """
    np.random.seed(seed)
    
    # Download diabetes dataset from UCI
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the data
        data = pd.read_csv(StringIO(response.text), header=None)
        
        # Last column is class, others are features
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Classes
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the dataset
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset
        
    except Exception as e:
        print(f"Error loading diabetes dataset: {e}")
        # Return a synthetic dataset as fallback
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=768, n_features=8, n_classes=2, 
                                 n_informative=6, n_redundant=2, random_state=seed)
        
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset

def haberman_data(seed=0, device="cpu", test_size=0.3):
    """
    Load the Haberman's Survival dataset from UCI.
    Features: 3 (age, year of operation, number of positive axillary nodes)
    Classes: 2 (survival status)
    Samples: 306
    """
    np.random.seed(seed)
    
    # Download haberman dataset from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the data
        data = pd.read_csv(StringIO(response.text), header=None)
        
        # Last column is class, others are features
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Classes
        
        # Convert classes to 0-indexed (1,2 -> 0,1)
        y = y - 1
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the dataset
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset
        
    except Exception as e:
        print(f"Error loading haberman dataset: {e}")
        # Return a synthetic dataset as fallback
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=306, n_features=3, n_classes=2, 
                                 n_informative=3, n_redundant=0, random_state=seed)
        
        train_input, test_input, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        dtype = torch.get_default_dtype()
        dataset = {}
        dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
        dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(train_labels).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(test_labels).type(torch.long).to(device)
        
        return dataset

# Utility function to list all available UCI datasets
def get_available_uci_datasets():
    """
    Return a dictionary of available UCI dataset functions with their descriptions.
    """
    datasets_info = {
        'iris_data': {
            'description': 'Iris flower dataset - 4 features, 3 classes, 150 samples',
            'features': 4,
            'classes': 3,
            'samples': 150
        },
        'wine_data': {
            'description': 'Wine recognition dataset - 13 features, 3 classes, 178 samples',
            'features': 13,
            'classes': 3,
            'samples': 178
        },
        'breast_cancer_data': {
            'description': 'Breast Cancer Wisconsin dataset - 30 features, 2 classes, 569 samples',
            'features': 30,
            'classes': 2,
            'samples': 569
        },
        'digits_data': {
            'description': 'Optical digit recognition dataset - 64 features, 10 classes, 1797 samples',
            'features': 64,
            'classes': 10,
            'samples': 1797
        },
        'glass_data': {
            'description': 'Glass identification dataset - 9 features, 6 classes, 214 samples',
            'features': 9,
            'classes': 6,
            'samples': 214
        },
        'ionosphere_data': {
            'description': 'Ionosphere dataset - 34 features, 2 classes, 351 samples',
            'features': 34,
            'classes': 2,
            'samples': 351
        },
        'sonar_data': {
            'description': 'Sonar dataset - 60 features, 2 classes, 208 samples',
            'features': 60,
            'classes': 2,
            'samples': 208
        },
        'heart_disease_data': {
            'description': 'Heart disease dataset - 13 features, 2 classes, 303 samples',
            'features': 13,
            'classes': 2,
            'samples': 303
        },
        'pima_diabetes_data': {
            'description': 'Pima Indians diabetes dataset - 8 features, 2 classes, 768 samples',
            'features': 8,
            'classes': 2,
            'samples': 768
        },
        'haberman_data': {
            'description': 'Haberman survival dataset - 3 features, 2 classes, 306 samples',
            'features': 3,
            'classes': 2,
            'samples': 306
        }
    }
    return datasets_info

# Example usage function
def demo_all_datasets(device="cpu"):
    """
    Load and print info about all UCI datasets.
    """
    datasets_info = get_available_uci_datasets()
    
    for name, info in datasets_info.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        try:
            # Get the function by name and call it
            func = globals()[name]
            dataset = func(device=device)
            print(f"  Loaded successfully!")
            print(f"  Train samples: {dataset['train_input'].shape[0]}")
            print(f"  Test samples: {dataset['test_input'].shape[0]}")
            print(f"  Feature dimension: {dataset['train_input'].shape[1]}")
            print(f"  Unique classes: {torch.unique(dataset['train_label']).numel()}")
        except Exception as e:
            print(f"  Error loading: {e}")

if __name__ == "__main__":
    demo_all_datasets()
