import argparse
from kan import *
import matplotlib.pyplot as plt
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from datasets import moon_data, random_data
from plotter import plot_train_data, plot_predictions, plot_violins, plot_violins_extended, plot_mean_std
from video import create_video



# SYMBOLIC FORMULA
def symbolic_regression(model, dataset):
    #if symb_reg:
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

    return train_acc_formula, test_acc_formula

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='KAN Model Training')
    
    # Utility
    parser.add_argument('--experiment_name', type=str, default="my_experiment", help='experiment name')
    parser.add_argument('--device_index', type=int, default=0, help='Grid size for the model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    # Model
    parser.add_argument('--hidden_width', type=int, default=3, help='Width of the hidden layers')
    parser.add_argument('--hidden_depth', type=int, default=1, help='Amount of the hidden layers')
    parser.add_argument('--steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--grid', type=int, default=5, help='Grid size for the model')
    parser.add_argument('--k', type=int, default=3, help='Parameter k for the model')
    parser.add_argument('--mode', type=str, choices=['default', 'abs', 'sigmoid', 'relu'], default='default', help='Activation mode')
    parser.add_argument('--base_fun', type=str, choices=['silu', 'identity', 'zero'], default='silu', help='base function')
    parser.add_argument('--spline_noise_scale', type=float, default=0.3, help='Adjust the spline noise at initialization')
    parser.add_argument('--init_mode', type=str, choices=['default', 'native_noise', 'width_in', 'width_out', 'xavier_in', 'xavier_out', 'xavier_torch'], default='default', help='Initialization Mode')
    #parser.add_argument('--native_noise_scale', type=bool, default=False, help='directly use the native spline_noise_scale value as std')

    # Trainable Features
    parser.add_argument('--sp_trainable', type=str2bool, default=False, help='Whether to make the spline parameters trainable')
    parser.add_argument('--sb_trainable', type=str2bool, default=False, help='Whether to make the spline basis trainable')
    parser.add_argument('--affine_trainable', type=str2bool, default=False, help='Whether to make affine parameters trainable')
    parser.add_argument('--update_grid', type=str2bool, default=False, help='Whether to update the grid during training')

    # Dataset
    parser.add_argument('--dataset', type=str, choices=['random', 'moon'], default='random', help='Select Dataset')
    parser.add_argument('--moon_noise_level', type=float, default=0, help='Adjust the noise for the moon dataset in the KAN')
    parser.add_argument('--random_distribution', type=str, choices=['uniform', 'normal'], default='random', help='Random Distribution')
    parser.add_argument('--random_input_dim', type=int, default=2, help='random Dataset Input Dimension')
    parser.add_argument('--random_output_dim', type=int, default=1, help='random Dataset Input Dimension')
    parser.add_argument('--random_uniform_range_min', type=float, default=-1, help='Random Uniform Dataset range min')
    parser.add_argument('--random_uniform_range_max', type=float, default=1, help='Random Uniform Dataset range max')
    parser.add_argument('--random_normal_mean', type=float, default=0, help='Random Normal Distribution Dataset mean')
    parser.add_argument('--random_normal_std', type=float, default=1, help='Random Normal Distribution Dataset std')

    # Eval & Plots
    parser.add_argument('--symbolic_regression', type=str2bool, default=False, help='Activates the Symbolic Regression. Takes long for big models')
    parser.add_argument('--plot_initialized_model', type=str2bool, default=False, help='Plot the initialized model (pykan native). Takes long and a lot of ram for big models')
    parser.add_argument('--plot_trained_model', type=str2bool, default=False, help='Plot the trained model (pykan native). Takes long and a lot of ram for big models')
    parser.add_argument('--save_video', type=str2bool, default=False, help='Save a video of the Splines (pykan native). Slows training')
    parser.add_argument('--save_model', type=str2bool, default=True, help='Save the model after training (can be very big if network is deep)')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args.experiment_name)
    device = torch.device(f'cuda:{args.device_index}' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # Start Experiment and run
    mlflow.set_experiment(args.experiment_name)
    run = mlflow.start_run()
    run_id = run.info.run_id

    # Save All Args in mlflow
    for arg, value in vars(args).items():
            mlflow.log_param(arg, value)

    # Get Dataset for training
    input_dim = None
    output_dim = None
    dataset=None
    if args.dataset == "random":
        #dataset = random_data()
        dataset =  random_data(
            args.random_distribution, 
            n_samples=10_000, 
            n_features=args.random_input_dim, 
            n_labels=args.random_output_dim, 
            loc=args.random_normal_mean,
            normal_scale=args.random_normal_std, 
            range=(args.random_uniform_range_min, args.random_uniform_range_max), 
            seed=args.seed,
            device=device
            )
        input_dim = args.random_input_dim
        output_dim = args.random_output_dim
    elif args.dataset == "moon":
        dataset = moon_data(
            data_noise_level=args.moon_noise_level, 
            n_samples=10_000, 
            seed=args.seed, 
            device=device
            )
        input_dim = 2
        output_dim = 1


    video_folder=f"./figures/{args.experiment_name}/{run_id}/video"
    ckpt_folder=f"./model/{args.experiment_name}/{run_id}"

    hidden = [args.hidden_width]*args.hidden_depth
    width = [input_dim, *hidden, output_dim]
    model = KAN(
            width=width, device=device,
            grid=args.grid, k=args.k, seed=args.seed,
            sp_trainable=args.sp_trainable, sb_trainable=args.sb_trainable, affine_trainable=args.affine_trainable,
            base_fun=args.base_fun,
            noise_scale=args.spline_noise_scale,
            mode=args.mode,
            init_mode=args.init_mode,
            ckpt_path=ckpt_folder
            )


    model(dataset['train_input'])

    print(f"dti: {dataset['train_input'].shape} dtl: {dataset['test_label'].shape}")
    ti = torch.round(model(dataset['train_input'])[:, 0])
    tl = dataset['test_label'][:, 0]
    print(f"ti: {ti.shape} tl: {tl.shape}")


    # Update plot_violins call
    fig = plot_violins(
        model=model, 
        sample_size=10_000, 
        title=f"Train Accuracy: Width: {args.hidden_width}, Init Mode: {args.init_mode}",
        mode='coef'
    )
    mlflow.log_figure(fig, "kan-coef-violins-initialized.png")
    fig = plot_violins(
        model=model, 
        sample_size=10_000, 
        title=f"Train Accuracy: Width: {args.hidden_width}, Init Mode: {args.init_mode}",
        mode='act'
    )
    mlflow.log_figure(fig, "kan-act-violins-initialized.png")
    # fig = plot_violins(
    #     model=model, 
    #     sample_size=10_000, 
    #     title=f"Train Accuracy: Width: {args.hidden_width}, Init Mode: {args.init_mode}",
    #     mode='grad'
    # )
    # mlflow.log_figure(fig, "kan-grad-violins-initialized.png")

    # Update plot_violins_extended call
    fig = plot_violins_extended(
        model=model, 
        dataset=dataset, 
        sample_size=100, 
        title=f"Train Accuracy: Width: {args.hidden_width}, Init Mode: {args.init_mode}"
    )
    mlflow.log_figure(fig, "kan-activations-violins-extended-initialized.png")

    # Update plot_mean_std call
    fig = plot_mean_std(
        model, 
        title=f"Train Accuracy: Width: {args.hidden_width}, Init Mode: {args.init_mode}"
    )
    mlflow.log_figure(fig, "layer_mean_std-initialized.png")

    #if args.plot_initialized_model:
    #if args.hidden_width < 10 and args.hidden_depth < 10 and args.random_input_dim < 10 and args.random_output_dim < 10:
    # Only plot small Networks
    if not any(element > 10 for sublist in width for element in sublist):
        model.plot(folder=f"./figures/{args.experiment_name}/{run_id}_initialized")
        mlflow.log_figure(model.fig, "kan-splines-initialized.png")


    # Metrics
    def train_acc():
        dtype = torch.get_default_dtype()
        return torch.mean((torch.round(model(dataset['train_input'])[:, 0]) == dataset['train_label'][:, 0]).type(dtype))

    def test_acc():
        dtype = torch.get_default_dtype()
        return torch.mean((torch.round(model(dataset['test_input'])[:, 0]) == dataset['test_label'][:, 0]).type(dtype))

    noises = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    indices = np.where(noises == args.spline_noise_scale)[0]
    spline_noise_scale_class = indices[0] if indices.size > 0 else -1
    mlflow.log_param("spline_noise_scale_class", spline_noise_scale_class)

    #results = model.fit(dataset, opt="LBFGS", steps=steps, metrics=(train_acc, test_acc, coef_mean, coef_std), update_grid=update_grid)
    results = model.fit(dataset, 
                        opt="LBFGS", 
                        steps=args.steps, 
                        metrics=(train_acc, test_acc), 
                        update_grid=args.update_grid,
                        img_folder=video_folder,
                        save_fig=args.save_video,
                        beta=10
                        )
    print(f"train_acc: {results['train_acc'][-1]:2f}, test_acc: {results['test_acc'][-1]:2f}")

    # Log Fitting Results in mlflow
    for i in range(len(results['train_acc'])):
        for key in results.keys():
            mlflow.log_metric(key, results[key][i], step=i)

    if args.save_model:
        mlflow.pytorch.log_model(model, "model")


    model(dataset['train_input'])
    # Update plot_violins call
    # Update plot_violins call
    fig = plot_violins(
        model=model, 
        sample_size=10_000, 
        title=f"Train Accuracy: {max(results['train_acc']):.2f}, Width: {args.hidden_width}, Init Mode: {args.init_mode}",
        mode='coef'
    )
    mlflow.log_figure(fig, "kan-coef-violins-trained.png")
    fig = plot_violins(
        model=model, 
        sample_size=10_000, 
        title=f"Train Accuracy: {max(results['train_acc']):.2f}, Width: {args.hidden_width}, Init Mode: {args.init_mode}",
        mode='act'
    )
    mlflow.log_figure(fig, "kan-act-violins-trained.png")
    fig = plot_violins(
        model=model, 
        sample_size=10_000, 
        title=f"Train Accuracy: {max(results['train_acc']):.2f}, Width: {args.hidden_width}, Init Mode: {args.init_mode}",
        mode='grad'
    )
    mlflow.log_figure(fig, "kan-grad-violins-trained.png")

    # Update plot_violins_extended call
    fig = plot_violins_extended(
        model=model, 
        dataset=dataset, 
        sample_size=100, 
        title=f"Train Accuracy: {max(results['train_acc']):.2f}, Width: {args.hidden_width}, Init Mode: {args.init_mode}"
    )
    mlflow.log_figure(fig, "kan-activations-violins-extended-trained.png")

    # Update plot_mean_std call
    fig = plot_mean_std(
        model, 
        title=f"Train Accuracy: {max(results['train_acc']):.2f}, Width: {args.hidden_width}, Init Mode: {args.init_mode}"
    )
    mlflow.log_figure(fig, "layer_mean_std-trained.png")


    #if args.plot_trained_model:
    #if args.hidden_width < 10 and args.hidden_depth < 10 and args.random_input_dim < 10 and args.random_output_dim < 10:
    # Only plot small Networks
    if not any(element > 10 for sublist in width for element in sublist):
        model.plot(folder=f"./figures/{args.experiment_name}/{run_id}_trained")
        mlflow.log_figure(model.fig, "kan-splines-trained.png")
    
    fig = plot_predictions(
        model=model, 
        dataset=dataset, 
        title=f"Trained Model - Train Accuracy: {max(results['train_acc']):.2f}, Width: {args.hidden_width}, Init Mode: {args.init_mode}"
    )
    mlflow.log_figure(fig, "test_input_predictions.png")

    # For trained plots
    fig = plot_train_data(
        dataset, 
        title=f"Trained Model - Train Accuracy: {max(results['train_acc']):.2f}, Width: {args.hidden_width}, Init Mode: {args.init_mode}"
    )
    mlflow.log_figure(fig, "train_data_trained.png")

    # SAVE Video
    if args.save_video:
        print("Save Video")
        video_name = "training"
        video_file = create_video(video_folder, video_name)
        mlflow.log_artifact(video_file)

    # Symbolic Refgression
    if args.symbolic_regression:
        train_acc_formula, test_acc_formula = symbolic_regression(model, dataset)
        print('train acc of the formula:', train_acc_formula)
        print('test acc of the formula:', test_acc_formula)

        mlflow.log_metric("train_acc_formula", train_acc_formula)
        mlflow.log_metric("test_acc_formula", test_acc_formula)

    mlflow.end_run()
    

if __name__ == "__main__":
    main()
