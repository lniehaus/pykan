import argparse
from kan import *
import matplotlib.pyplot as plt
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from datasets import random_data, moon_data, mnist_data, cifar10_data, make_classification_data, mnist1d_data, boxes_2d_dataset
from plotter import plot_train_data, plot_predictions, plot_violins, plot_violins_extended, plot_summed_violins, plot_mean_std, plot_layerwise_postacts_and_postsplines, generate_grid_tensor, plot_decision_boundary, plot_classifier_probes
from video import create_video
from metrics import count_connected_regions, calc_boundary_length, calc_boundary_curvature, calc_fractal_dimension

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
    parser.add_argument('--hidden_form', type=str, choices=['square', 'linear', 'kat'], default='square', help='Architecture mode')
    parser.add_argument('--hidden_width', type=int, default=3, help='Width of the hidden layers')
    parser.add_argument('--hidden_depth', type=int, default=1, help='Amount of the hidden layers')
    parser.add_argument('--steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--grid', type=int, default=5, help='Grid size for the model')
    parser.add_argument('--k', type=int, default=3, help='Parameter k for the model')
    parser.add_argument('--mode', type=str, choices=['default', 'abs', 'sigmoid', 'relu'], default='default', help='Activation mode')
    parser.add_argument('--base_fun', type=str, choices=['silu', 'identity', 'zero'], default='silu', help='base function')
    parser.add_argument('--spline_noise_scale', type=float, default=0.3, help='Adjust the spline noise at initialization')
    parser.add_argument('--init_mode', type=str, choices=['default', 'default-0_1', 'default-0_3', 'default-0_5', 'native_noise', 'width_in', 'width_out', 'xavier_in', 'xavier_out', 'xavier_torch', 'width_in_num', 'xavier_in_num', 'width_in_out', 'xavier_in_out', 'width_in_out_num', 'xavier_in_out_num', 'kaiming_in', 'kaiming_in_out', 'kaiming_leaky_in', 'kaiming_leaky_in_out'], default='default', help='Initialization Mod. default=use spline_noise_scale parameter, default-0_1=use sns 0.1, default-0_3=use sns 0.3, default-0_5=use sns 0.5')
    #parser.add_argument('--native_noise_scale', type=bool, default=False, help='directly use the native spline_noise_scale value as std')
    parser.add_argument('--grid_mode', type=str, choices=['default', 'native', 'xavier', 'xavier_10', 'xavier_x'], default='default', help='Grid Range Mode. default=use grid_range. xavier_x uses the grid_bound to scale the xavier range.')
    parser.add_argument('--grid_bound', type=float, default=1, help='If grid_mode is set to native, use this value for the bounds of the grid. default=1.0')   
    parser.add_argument('--learning_rate', type=float, default=1, help='Learning Rate for the optimizer')   
    parser.add_argument('--lamb', type=float, default=0.0, help='Weight decay for the optimizer')
    parser.add_argument('--lamb_l1', type=float, default=1.0, help='Weight for the L1 loss')
    parser.add_argument('--lamb_entropy', type=float, default=0.0, help='Weight for the entropy loss')
    parser.add_argument('--lamb_coef', type=float, default=0.0, help='Weight for the coefficient loss')
    parser.add_argument('--lamb_coefdiff', type=float, default=0.0, help='Weight for the coefficient difference loss')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'LBFGS'], default='LBFGS', help='Optimizer for training')

    # Trainable Features
    parser.add_argument('--sp_trainable', type=str2bool, default=False, help='Whether to make the spline parameters trainable')
    parser.add_argument('--sb_trainable', type=str2bool, default=False, help='Whether to make the spline basis trainable')
    parser.add_argument('--affine_trainable', type=str2bool, default=False, help='Whether to make affine parameters trainable')
    parser.add_argument('--update_grid', type=str2bool, default=False, help='Whether to update the grid during training')

    # Dataset
    parser.add_argument('--dataset', type=str, choices=['random', 'moon', 'mnist', 'cifar10', 'make_classification', 'mnist1d', 'boxes_2d'], default='random', help='Select Dataset')
    parser.add_argument('--moon_noise_level', type=float, default=0, help='Adjust the noise for the moon dataset in the KAN')
    parser.add_argument('--random_distribution', type=str, choices=['uniform', 'normal'], default='random', help='Random Distribution')
    parser.add_argument('--random_input_dim', type=int, default=2, help='random Dataset Input Dimension')
    parser.add_argument('--random_output_dim', type=int, default=1, help='random Dataset Input Dimension')
    parser.add_argument('--random_uniform_range_min', type=float, default=-1, help='Random Uniform Dataset range min')
    parser.add_argument('--random_uniform_range_max', type=float, default=1, help='Random Uniform Dataset range max')
    parser.add_argument('--random_normal_mean', type=float, default=0, help='Random Normal Distribution Dataset mean')
    parser.add_argument('--random_normal_std', type=float, default=1, help='Random Normal Distribution Dataset std')
    parser.add_argument('--mnist1d_subset_size', type=int, default=100_000, help='Subset size for the mnist1d dataset') 
    parser.add_argument('--boxes_n_classes', type=int, default=4, help='Number of classes for the boxes 2d dataset')
    parser.add_argument('--boxes_datapoints_per_class', type=int, default=10, help='Number of datapoints per class for the boxes 2d dataset')


    # Eval & Plots
    parser.add_argument('--symbolic_regression', type=str2bool, default=False, help='Activates the Symbolic Regression. Takes long for big models')
    parser.add_argument('--plot_initialized_model', type=str2bool, default=False, help='Plot the initialized model (pykan native). Takes long and a lot of ram for big models')
    parser.add_argument('--plot_trained_model', type=str2bool, default=False, help='Plot the trained model (pykan native). Takes long and a lot of ram for big models')
    parser.add_argument('--save_video', type=str2bool, default=False, help='Save a video of the Splines (pykan native). Slows training')
    parser.add_argument('--save_model', type=str2bool, default=True, help='Save the model after training (can be very big if network is deep)')

    args = parser.parse_args()
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    #torch.set_default_dtype(torch.float64)
    #torch.set_default_dtype(torch.float16)

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

    # set spline noise scale
    if args.init_mode == "default":
        args.spline_noise_scale = args.spline_noise_scale
    elif args.init_mode == "default-0_1":
        args.spline_noise_scale = 0.1
    elif args.init_mode == "default-0_3":
        args.spline_noise_scale = 0.3
    elif args.init_mode == "default-0_5":
        args.spline_noise_scale = 0.5

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
            #n_features=2, 
            #n_labels=2, 
            seed=args.seed, 
            device=device
            )
        input_dim = 2
        output_dim = 2
    elif args.dataset == "mnist":
        dataset = mnist_data(device=device)
        input_dim = 784
        output_dim = 10
    elif args.dataset == "cifar10":
        dataset = cifar10_data(device=device, subset_size=100_000, grayscale=False)
        #input_dim = 1024 # grayscale
        input_dim = 3072 # rgb
        output_dim = 10
    elif args.dataset == "make_classification":
        dataset = make_classification_data(n_samples=1000, 
                                           n_features=args.random_input_dim, 
                                           n_labels=args.random_output_dim, 
                                           n_informative=2, 
                                           n_redundant=0, 
                                           seed=args.seed, 
                                           device=device)
        #input_dim = 1024 # grayscale
        input_dim = args.random_input_dim # rgb
        output_dim = args.random_output_dim
    elif args.dataset == "mnist1d":
        dataset = mnist1d_data(device=device, seed=args.seed, subset_size=100_000)
        input_dim = 40
        output_dim = 10
    elif args.dataset == "boxes_2d":
        dataset = boxes_2d_dataset(
            n_classes=args.boxes_n_classes,
            datapoints_per_class=args.boxes_datapoints_per_class,
            bounds=(args.random_uniform_range_min, args.random_uniform_range_max, args.random_uniform_range_min, args.random_uniform_range_max),
            device=device
        )
        input_dim = 2
        output_dim = args.boxes_n_classes


    video_folder=f"./figures/{args.experiment_name}/{run_id}/video"
    ckpt_folder=f"./model/{args.experiment_name}/{run_id}"

    hidden = [args.hidden_width]*args.hidden_depth
    width = [input_dim, *hidden, output_dim]

    #hidden_form = "square"
    width = []
    hidden_widths = []
    if args.hidden_form == "square":
        hidden_widths = [args.hidden_width]*args.hidden_depth
        #width = [input_dim, *hidden_widths, output_dim]
    elif args.hidden_form == "linear":
        # Create a list of widths that interpolate from input_dim to output_dim
        if args.hidden_depth > 0:
            # Generate linearly spaced values between input_dim and output_dim
            hidden_widths = [int(x) for x in np.linspace(input_dim, output_dim, args.hidden_depth + 2)[1:-1]]
        else:
            hidden_widths = []
        # Combine input_dim, hidden widths, and output_dim
        #width = [input_dim, *hidden_widths, output_dim]
    elif args.hidden_form == "kat":
        # Create a list of widths that interpolate from input_dim to output_dim
        first_layer = int(2*input_dim) + 1
        if args.hidden_depth > 0:
            # Generate linearly spaced values between input_dim and output_dim
            hidden_widths = [int(x) for x in np.linspace(first_layer, output_dim, args.hidden_depth + 1)[0:-1]]
        else:
            hidden_widths = []
        #width = [input_dim, *hidden_widths, output_dim]
        #print("width", width)
    else:
        raise ValueError("hidden_form must be either 'square', 'linear' or 'kat'")

    #print("hidden_widths", hidden_widths)
    width = [input_dim, *hidden_widths, output_dim]
    print("width", width)

    model = KAN(
            width=width, device=device,
            grid=args.grid, k=args.k, seed=args.seed,
            sp_trainable=args.sp_trainable, sb_trainable=args.sb_trainable, affine_trainable=args.affine_trainable,
            base_fun=args.base_fun,
            noise_scale=args.spline_noise_scale,
            mode=args.mode,
            init_mode=args.init_mode,
            grid_mode=args.grid_mode,
            grid_bound=args.grid_bound,
            ckpt_path=ckpt_folder
            )

    parameters = count_parameters(model)
    mlflow.log_metric("parameters", parameters)

    model(dataset['train_input'])

    fig = plot_layerwise_postacts_and_postsplines(
        model=model,
        title=f"Layerwise Postacts & Postsplines - Width: {args.hidden_width}, Init Mode: {args.init_mode}"
    )
    mlflow.log_figure(fig, "layerwise_postacts_and_postsplines-initialized.png")

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

    # # Update plot_violins_extended call
    # fig = plot_violins_extended(
    #     model=model, 
    #     dataset=dataset, 
    #     sample_size=100, 
    #     title=f"Train Accuracy: Width: {args.hidden_width}, Init Mode: {args.init_mode}"
    # )
    # mlflow.log_figure(fig, "kan-activations-violins-extended-initialized.png")

    fig = plot_summed_violins(
        model=model, 
        sample_size=10_000, 
        title=f"Train Accuracy: Width: {args.hidden_width}, Init Mode: {args.init_mode}",
        mode='act'
    )
    mlflow.log_figure(fig, "kan-act-summed-violins-initialized.png")

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
        model.plot(scale=1.0, folder=f"./figures/{args.experiment_name}/{run_id}_initialized", beta=100)
        mlflow.log_figure(model.fig, "kan-splines-initialized.png")

    def train_acc():
        return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

    def test_acc():
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

    noises = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    indices = np.where(noises == args.spline_noise_scale)[0]
    spline_noise_scale_class = indices[0] if indices.size > 0 else -1
    mlflow.log_param("spline_noise_scale_class", spline_noise_scale_class)

    metrics = (train_acc, test_acc)
    

    #results = model.fit(dataset, opt="LBFGS", steps=steps, metrics=(train_acc, test_acc, coef_mean, coef_std), update_grid=update_grid)
    results = model.fit(dataset, 
                        opt=args.optimizer, 
                        steps=args.steps, 
                        metrics=metrics, 
                        update_grid=args.update_grid,
                        img_folder=video_folder,
                        save_fig=args.save_video,
                        loss_fn=torch.nn.CrossEntropyLoss(),
                        lr=args.learning_rate,
                        lamb=args.lamb,
                        lamb_l1=args.lamb_l1,
                        lamb_entropy=args.lamb_entropy,
                        lamb_coef=args.lamb_coef,
                        lamb_coefdiff=args.lamb_coefdiff
                        )
    print(f"train_acc: {results['train_acc'][-1]:2f}, test_acc: {results['test_acc'][-1]:2f}")

    # Log Fitting Results in mlflow
    for i in range(len(results['train_acc'])):
        for key in results.keys():
            mlflow.log_metric(key, results[key][i], step=i)


    mlflow.log_metric("train_acc_max", max(results['train_acc']), step=0)
    mlflow.log_metric("test_acc_max", max(results['test_acc']), step=0)

    if args.save_model:
        mlflow.pytorch.log_model(model, "model")

    # classifier probes
    #from sklearn.linear_model import SGDClassifier
    model.eval()

    model(dataset['train_input'])

    fig = plot_layerwise_postacts_and_postsplines(
        model,
        title=f"Layerwise Postacts & Postsplines - Train Acc: {max(results['train_acc']):.2f}, Width: {args.hidden_width}, Init Mode: {args.init_mode}"
    )
    mlflow.log_figure(fig, "layerwise_postacts_and_postsplines-trained.png")

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

    # # Update plot_violins_extended call
    # fig = plot_violins_extended(
    #     model=model, 
    #     dataset=dataset, 
    #     sample_size=100, 
    #     title=f"Train Accuracy: {max(results['train_acc']):.2f}, Width: {args.hidden_width}, Init Mode: {args.init_mode}"
    # )
    # mlflow.log_figure(fig, "kan-activations-violins-extended-trained.png")

    fig = plot_summed_violins(
        model=model, 
        sample_size=10_000, 
        title=f"Train Accuracy: Width: {args.hidden_width}, Init Mode: {args.init_mode}",
        mode='act'
    )
    mlflow.log_figure(fig, "kan-act-summed-violins-trained.png")

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
        model.plot(scale=1.0, folder=f"./figures/{args.experiment_name}/{run_id}_trained", beta=100)
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

    # # # Train Accuracy Classifier Probe
    # # model(dataset['train_input'])
    # # targets = dataset['train_label'].cpu().detach().numpy()
    # # for i, postacts in enumerate(model.spline_postacts):
    # #     postacts_np = postacts.cpu().detach().numpy()
    # #     postacts_np = postacts_np.reshape(postacts_np.shape[0], -1)
    # #     #print("data shape",postacts_np.shape)
    # #     classifier = SGDClassifier(penalty=None, loss="log_loss", learning_rate="constant", eta0=0.01)
    # #     classifier.fit(postacts_np, targets)
    # #     score = classifier.score(postacts_np, targets)
    # #     name = f"classifier_probe_train_accuracy"
    # #     print("Train Classifier Probe", name, i, score)
    # #     mlflow.log_metric(name, score, step=i)

    # # Train Accuracy Classifier Probe
    # model(dataset['train_input'])
    # targets = dataset['train_label'].cpu().detach().numpy()
    # for i, preacts in enumerate(model.spline_preacts):
    #     preacts_np = preacts.cpu().detach().numpy()
    #     preacts_np = preacts_np.reshape(preacts_np.shape[0], -1)
    #     #print("data shape",postacts_np.shape)
    #     classifier = SGDClassifier(penalty=None, loss="log_loss", learning_rate="constant", eta0=0.01)
    #     classifier.fit(preacts_np, targets)
    #     score = classifier.score(preacts_np, targets)
    #     name = f"classifier_probe_train_accuracy"
    #     print("Train Classifier Probe", name, i, score)
    #     mlflow.log_metric(name, score, step=i)


    # # # Test Accuracy Classifier Probe
    # # model(dataset['test_input'])
    # # targets = dataset['test_label'].cpu().detach().numpy()
    # # for i, postacts in enumerate(model.spline_postacts):
    # #     postacts_np = postacts.cpu().detach().numpy()
    # #     postacts_np = postacts_np.reshape(postacts_np.shape[0], -1)
    # #     #print("data shape",postacts_np.shape)
    # #     classifier = SGDClassifier(penalty=None, loss="log_loss", learning_rate="constant", eta0=0.01)
    # #     classifier.fit(postacts_np, targets)
    # #     score = classifier.score(postacts_np, targets)
    # #     name = f"classifier_probe_test_accuracy"
    # #     print("Test Classifier Probe", name, i, score)
    # #     mlflow.log_metric(name, score, step=i)

    # # Test Accuracy Classifier Probe
    # model(dataset['test_input'])
    # targets = dataset['test_label'].cpu().detach().numpy()
    # for i, preacts in enumerate(model.spline_preacts):
    #     preacts_np = preacts.cpu().detach().numpy()
    #     preacts_np = preacts_np.reshape(preacts_np.shape[0], -1)
    #     #print("data shape",postacts_np.shape)
    #     classifier = SGDClassifier(penalty=None, loss="log_loss", learning_rate="constant", eta0=0.01)
    #     classifier.fit(preacts_np, targets)
    #     score = classifier.score(preacts_np, targets)
    #     name = f"classifier_probe_test_accuracy"
    #     print("Test Classifier Probe", name, i, score)
    #     mlflow.log_metric(name, score, step=i)


    if args.dataset == 'boxes_2d' or args.dataset == 'moon' or args.random_input_dim == 2:
        grid_tensor, xx, yy = generate_grid_tensor(bounds=(-1, 1, -1, 1), resolution=1000, device=device, dtype=torch.FloatTensor)

        fig = plot_decision_boundary(
            model=model,
            dataset=dataset,
            grid_tensor=grid_tensor,
            xx=xx,
            yy=yy,
            title=f"Decision Boundary - Train Acc: {results['train_acc'][-1]:.2f}, Test Acc: {results['test_acc'][-1]:.2f}"
        )
        mlflow.log_figure(fig, "decision_boundary.png")

        # Calculate Decision Boundary Metrics
        model.eval()
        with torch.no_grad():
            pred = model(grid_tensor).argmax(dim=1).cpu().numpy().reshape(xx.shape)
        connected_regions = count_connected_regions(pred)
        boundary_length = calc_boundary_length(pred, xx, yy)
        curvature = calc_boundary_curvature(pred, xx, yy)
        fractal_dim = calc_fractal_dimension(pred)
        print(f"Connected Regions: {connected_regions}, Boundary Length: {boundary_length}, Curvature: {curvature}, Fractal Dimension: {fractal_dim}")
        mlflow.log_metric("connected_regions", connected_regions)
        mlflow.log_metric("boundary_length", boundary_length)
        mlflow.log_metric("curvature", curvature)
        mlflow.log_metric("fractal_dimension", fractal_dim)


    fig = plot_classifier_probes(
        model, 
        dataset=dataset,
        evalset='train',
        title="Layerwise Classifier Probes Train evaluation"
    )
    mlflow.log_figure(fig, "classifier_probes_train.png")

    fig = plot_classifier_probes(
        model, 
        dataset=dataset,
        evalset='test',
        title="Layerwise Classifier Probes Test evaluation"
    )
    mlflow.log_figure(fig, "classifier_probes_test.png")

    mlflow.end_run()
    

if __name__ == "__main__":
    main()
