import argparse
from kan import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np
import mlflow
import mlflow.pytorch
import torch
from video import create_video

# Set up argument parser
parser = argparse.ArgumentParser(description='KAN Model Training')
parser.add_argument('--experiment_name', type=str, default="my_experiment", help='experiment name')
#parser.add_argument('--base_width', type=int, nargs='+', default=[5], help='Base width of the model layers')
parser.add_argument('--hidden_width', type=int, default=3, help='Width of the hidden layers')
parser.add_argument('--hidden_depth', type=int, default=1, help='Amount of the hidden layers')
parser.add_argument('--steps', type=int, default=100, help='Number of training steps')
parser.add_argument('--grid', type=int, default=5, help='Grid size for the model')
parser.add_argument('--k', type=int, default=3, help='Parameter k for the model')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--sp_trainable', type=bool, default=False, help='Whether to make the spline parameters trainable')
parser.add_argument('--sb_trainable', type=bool, default=False, help='Whether to make the spline basis trainable')
parser.add_argument('--affine_trainable', type=bool, default=False, help='Whether to make affine parameters trainable')
parser.add_argument('--update_grid', type=bool, default=False, help='Whether to update the grid during training')
parser.add_argument('--mode', type=str, choices=[None, 'abs', 'sigmoid', 'relu'], default=None, help='Activation mode')
parser.add_argument('--base_fun', type=str, choices=['silu', 'identity', 'zero'], default='silu', help='Activation mode')
parser.add_argument('--data_noise_level', type=float, default=0, help='Adjust the noise in the KAN')
parser.add_argument('--spline_noise_scale', type=float, default=0.3, help='Adjust the spline noise at initialization')
parser.add_argument('--native_noise_scale', type=bool, default=False, help='directly use the native spline_noise_scale value as std')
parser.add_argument('--device_index', type=int, default=0, help='Grid size for the model')

args = parser.parse_args()




# HYPERPARAMETERS

#base_width = args.base_width
hidden_width = args.hidden_width
hidden_depth = args.hidden_depth
steps = args.steps
grid = args.grid
k = args.k
seed = args.seed
sp_trainable = args.sp_trainable
sb_trainable = args.sb_trainable
affine_trainable = args.affine_trainable
update_grid = args.update_grid
data_noise_level = args.data_noise_level
spline_noise_scale = args.spline_noise_scale
native_noise_scale = args.native_noise_scale
experiment_name = args.experiment_name
mode = args.mode
base_fun = args.base_fun
device_index = args.device_index

# Get Noise Class
noises = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])
indices = np.where(noises == spline_noise_scale)[0]
spline_noise_scale_class = indices[0] if indices.size > 0 else -1

# plot_model = False
# symb_reg = False

plot_model = True
symb_reg = True

device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')
print(device)

mlflow.set_experiment(experiment_name)
print(experiment_name)

# DATASET

# dataset = {}
# train_input, train_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)
# test_input, test_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)

# dtype = torch.get_default_dtype()
# dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
# dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
# dataset['train_label'] = torch.from_numpy(train_label[:, None]).type(dtype).to(device)
# dataset['test_label'] = torch.from_numpy(test_label[:, None]).type(dtype).to(device)

# X = dataset['train_input']
# y = dataset['train_label']
# plt.scatter(X[:, 0].cpu().detach().numpy(), X[:, 1].cpu().detach().numpy(), c=y[:, 0].cpu().detach().numpy())


# Generate the original dataset
train_input, train_label = make_moons(n_samples=1000, shuffle=True, noise=data_noise_level, random_state=seed)
test_input, test_label = make_moons(n_samples=1000, shuffle=True, noise=data_noise_level, random_state=seed+1)

# Convert to PyTorch tensors
dtype = torch.get_default_dtype()
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = {}
dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
dataset['train_label'] = torch.from_numpy(train_label[:, None]).type(dtype).to(device)
dataset['test_label'] = torch.from_numpy(test_label[:, None]).type(dtype).to(device)

# # Function to add noise
# def add_noise(data, noise_level=0.1):
#     noise = np.random.normal(0, data_noise_level, data.shape)  # Generate Gaussian noise
#     return data + noise

# # Add noise to the training and test inputs
# #noise_level = 0.1 # Adjust the noise level as needed
# dataset['train_input'] = add_noise(dataset['train_input'].cpu().numpy(), data_noise_level)
# dataset['test_input'] = add_noise(dataset['test_input'].cpu().numpy(), data_noise_level)

# # Convert back to PyTorch tensors
# dataset['train_input'] = torch.from_numpy(dataset['train_input']).type(dtype).to(device)
# dataset['test_input'] = torch.from_numpy(dataset['test_input']).type(dtype).to(device)

# Visualize the noisy training data
X = dataset['train_input']
y = dataset['train_label']
plt.scatter(X[:, 0].cpu().detach().numpy(), X[:, 1].cpu().detach().numpy(), c=y[:, 0].cpu().detach().numpy())
plt.title('Noisy Moons Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

run = mlflow.start_run()
run_id = run.info.run_id
mlflow.log_figure(plt.gcf(), "train_data.png")
#plt.show()



# MODEL

hidden = [hidden_width]*hidden_depth
width = [2, *hidden, 1]
#width = [2, *base_width, 1]
model = KAN(width=width, grid=grid, k=k, device=device, seed=seed,
            sp_trainable=sp_trainable, sb_trainable=sb_trainable, affine_trainable=affine_trainable,
            base_fun=base_fun,
            noise_scale=spline_noise_scale,
            mode=mode,
            native_noise_scale=native_noise_scale
            )

model(dataset['train_input'])

if plot_model:
    model.plot(beta=100, folder=f"./figures/{experiment_name}/{run_id}")
    mlflow.log_figure(model.fig, "kan-splines-initialization.png")

def train_acc():
    return torch.mean((torch.round(model(dataset['train_input'])[:, 0]) == dataset['train_label'][:, 0]).type(dtype))

def test_acc():
    return torch.mean((torch.round(model(dataset['test_input'])[:, 0]) == dataset['test_label'][:, 0]).type(dtype))

def coef_mean():
    all_coefs = []
    for i in range(len(model.width)-1):
        all_coefs.append(model.act_fun[i].coef)
    all_coefs_tensor = torch.cat([coef.view(-1) for coef in all_coefs])
    return all_coefs_tensor.mean()

def coef_std():
    all_coefs = []
    for i in range(len(model.width)-1):
        all_coefs.append(model.act_fun[i].coef)
    all_coefs_tensor = torch.cat([coef.view(-1) for coef in all_coefs])
    return all_coefs_tensor.std()

# Start MLflow run

mlflow.log_param("mode", mode)
mlflow.log_param("data_noise_level", data_noise_level)
mlflow.log_param("spline_noise_scale", spline_noise_scale)
mlflow.log_param("spline_noise_scale_log", np.log(1+spline_noise_scale))
mlflow.log_param("spline_noise_scale_class", spline_noise_scale_class)
mlflow.log_param("native_noise_scale", native_noise_scale)
mlflow.log_param("hidden_width", hidden_width)
mlflow.log_param("hidden_depth", hidden_depth)
mlflow.log_param("width", width)
mlflow.log_param("steps", steps)
mlflow.log_param("grid", grid)
mlflow.log_param("k", k)
mlflow.log_param("sp_trainable", sp_trainable)
mlflow.log_param("sb_trainable", sb_trainable)
mlflow.log_param("affine_trainable", affine_trainable)
mlflow.log_param("update_grid", update_grid)
mlflow.log_param("seed", seed)

video_folder=f"./figures/{experiment_name}/{run_id}/video"

#results = model.fit(dataset, opt="LBFGS", steps=steps, metrics=(train_acc, test_acc, coef_mean, coef_std), update_grid=update_grid)
results = model.fit(dataset, 
                    opt="LBFGS", 
                    steps=steps, 
                    metrics=(train_acc, test_acc), 
                    update_grid=update_grid,
                    img_folder=video_folder,
                    save_fig=True,
                    beta=10
                    )
print(results['train_acc'][-1], results['test_acc'][-1])

for i in range(len(results['train_acc'])):
    mlflow.log_metric("train_acc", results['train_acc'][i], step=i)
    mlflow.log_metric("test_acc", results['test_acc'][i], step=i)
    mlflow.log_metric("train_loss", results['train_loss'][i], step=i)
    mlflow.log_metric("test_loss", results['test_loss'][i], step=i)
    mlflow.log_metric("reg", results['reg'][i], step=i)
    #mlflow.log_metric("coef_mean", results['coef_mean'][i], step=i)
    #mlflow.log_metric("coef_std", results['coef_std'][i], step=i)

mlflow.pytorch.log_model(model, "model")

# SAVE Video

print("Save Video")
video_name = "training"
create_video(video_folder)
video_file = os.path.join(video_folder, video_name+'.mp4')
mlflow.log_artifact(video_file)


# KAN SPLINES

if plot_model:
    print("Plot Model")
    model.plot(beta=100, folder=f"./figures/{experiment_name}/{seed}")
    mlflow.log_figure(model.fig, "kan-splines-trained.png")

# PREDICTIONS

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
mlflow.log_figure(plt.gcf(), "test_input_predictions.png")

# SYMBOLIC FORMULA

if symb_reg:
    print("Symbolic Regression")
    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs']
    model.auto_symbolic(lib=lib)
    formula = model.symbolic_formula()[0][0]
    ex_round(formula, 4)

    # how accurate is this formula?
    def acc(formula, X, y):
        batch = X.shape[0]
        correct = 0
        for i in range(batch):
            correct += np.round(np.array(formula.subs('x_1', X[i, 0]).subs('x_2', X[i, 1])).astype(np.float64)) == y[i, 0]
        return correct / batch

    train_acc_formula = acc(formula, dataset['train_input'], dataset['train_label'])
    test_acc_formula = acc(formula, dataset['test_input'], dataset['test_label'])
    print('train acc of the formula:', train_acc_formula)
    print('test acc of the formula:', test_acc_formula)

    mlflow.log_metric("train_acc_formula", train_acc_formula)
    mlflow.log_metric("test_acc_formula", test_acc_formula)

mlflow.end_run()