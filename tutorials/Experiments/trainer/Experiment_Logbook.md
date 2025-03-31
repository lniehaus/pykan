
# Results of width_15

- xavier_in highest train accuracy: 0.877299964427948
- width_in highest train accuracy: 0.6563000082969666
- default highest train accuracy: 0.5072000026702881

xavier_in allows to train compared to default, which does not.
everything is off except coef training
- Why is xavier_in better?
- Does xavier show more overfitting tendencies? 

width_in shows to be in between the xavier_in and default.
width_in has 2 runs achieving more than 0.5 train accurarcy (width: 10, 20).
for xavier all 10 runs achieved more tha 0.5 train accuracy (width: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100).
In default none of the runs achieved marginally more that 0.5 train accuracy.
The width describes the random architecture.
all runs have a depth of 5.
the width describes the width for each hidden layer and also the amount of input features and output targets.
The Architecture is like a Rectangle (5, width+2).
+2 because of the input and output layer.
The task is a regression with random input data and random output labels.
Compared to other datasets (make_moons?) a random dataset allows a dynamic range for the input and output of the Architecture and Dataset.
By this the Network can always learn a task, where the task is harder when the network gets wider.
Main Argument: Wider and Deeper KANs can approximate more data.

future:
- How about a run with dynamic depth?
- this should result in something different because we do not scale the complexity of the task like we do it when we scale the width.


# Results of width_dataset_init_03 
stopped early.
Contains only some big dataset runs
However, we can inspect the Networks and their performances for width 100
and for the initialization mode: 'default' 'native_noise' 'width_in' 'xavier_in' 'xavier_torch'.

| Rank | Init Mode        | Train Accuracy |
|------|------------------|----------------|
| 1    | native_noise 0.3 | 0.888799965    |
| 2    | xavier_in        | 0.870399951    |
| 3    | default          | 0.50309997     |
| 3    | width_in         | 0.50309997     |
| 4    | xavier_torch     | 0.152899995    |

xavier_in and native_noise 0.3 work good.
default is the same as width_in.
and xavier_torch is the exact opposite from what the task asks (1-0.15= 75%).
Since it is a binary task this could be positive as well.
We further inspect the kan-activations-violins-trained.png

## xavier_torch: 0.15%
http://127.0.0.1:5040/#/experiments/668307255334764331/runs/3c5e0ace3e344c6d831537dd3001f2ce/artifacts
Activations through the layers mostly stable.
in the trained violins activation plot the xavier_torch activations are very similar through the network (depth=5). The y (postacts) output of the layer is the input for the next layer x (preacts).
The activations reach outside of the grid (-1,1)
- Their Kurtosis seems to be squite strong.
- Maybe a lot of outliers?
- are the outliers importat? (feature importance? score that influences the opacity of the individual splines)
- measure the skewness and kurtosis in addition to mean and std

http://127.0.0.1:5040/#/experiments/668307255334764331/runs/3c5e0ace3e344c6d831537dd3001f2ce/artifacts
The extended plot shows that a lot of activations are very close to 0 (or exactly?) (laplace distribution?)

## width_in: 50%
http://127.0.0.1:5040/#/experiments/668307255334764331/runs/efbb79d29ddf478b9f38cf0ac3d9f4fe/artifacts
Activations through the layers not stable.
The last layer is an outlier, with a mean of ~+0.005, while the others mean is around 0.0. 
http://127.0.0.1:5040/#/experiments/668307255334764331/runs/efbb79d29ddf478b9f38cf0ac3d9f4fe/artifacts
activations are gaussian like distributed.

## default: 50%
http://127.0.0.1:5040/#/experiments/668307255334764331/runs/6e0619b4e8c24f5ca8b91ff99ebfc5dc/artifacts
Activations through the layers not stable.
The last layer is an outlier, with a mean of ~+0.005, while the others mean is around 0.0. 
http://127.0.0.1:5040/#/experiments/668307255334764331/runs/6e0619b4e8c24f5ca8b91ff99ebfc5dc/artifacts
activations are gaussian like distributed.

## xavier_in: 87%
http://127.0.0.1:5040/#/experiments/668307255334764331/runs/cc8dbf498d924574826ce5cbcf710e79/artifacts
Activations through the layers mostly stable.
the activation range us inside the grid (-1,1) with roughtly (-0.2, 0.2)
http://127.0.0.1:5040/#/experiments/668307255334764331/runs/cc8dbf498d924574826ce5cbcf710e79/artifacts
activations are gaussian like distributed.

## native_noise 0.3: 88%
http://127.0.0.1:5040/#/experiments/668307255334764331/runs/2dd7037ebb834b7a8085cfd838969242/artifacts
Activations through the layers mostly stable.
the activation range us inside the grid (-1,1) with roughtly (-0.2, 0.2)
http://127.0.0.1:5040/#/experiments/668307255334764331/runs/2dd7037ebb834b7a8085cfd838969242/artifacts
activations are gaussian like distributed.

## Conclusion
native_noise 0.3: 88% is not only close in accuracy (xavier_in: 87%), but also in trained activations range inside the grid. They are zero centered with a stable std through the layers.

default: 50% and width_in: 50% also show similar activation behaviour between each other. They a re not stablethrough the layers.
Their means are roughly at the same position for each layer. all layers at 0.0, except for the last layer, which has a mean of ~+0.005.
Their variance distribution seems slightly different. default seems to have a larger kurtosis than width_in.

# Results of width_dataset_init_04
This experiment compares the random and moon datasets with differing width sizes.
xavier_torch shows strange behaviour.

## xavier_torch observations
For the moon dataset it reaches a train accuracy between 69% (w:10) to 99% (w100).
However, the test accuracy on moon is between 40% (w:50) to 49% (w:60, w:100).
Since moon contains actual trainable information we know, that xavier_torch results in strong overfitting.
For the random dataset on the other hand, the train accuracy is between 16% (w:100) and 29% (w:10).
The test accuracy is between 13% (w:100) and 26% (w:10)
- Why is the test accuracy for random not around 50%?
- A test accuracy of (100%-13%=87%) on binary random data is crazy good
- There seems to be a bug regarding the generation of labels with a dynamic size
future
- moon dataset goes currently from -3 to +3, which is outside of the defined grid (-1,+1)
- test this with a normalized version of moon (z-rescaling, or min-max-rescaling).

## xavier_in

## width_in

## native_noise 0.3

## default

# Results of depth_dataset_init_01
Training depth takes much longer than training width.


# Sources (need to read)
- Kolmogorov-Arnold Networks with Trainable
Activation Functions for Data Regression and
Classification
file:///home/luniehaus/Downloads/1571098990.pdf

- A Benchmarking Study of Kolmogorov-Arnold
Networks on Tabular Data
https://arxiv.org/pdf/2406.14529

- Kolmogorov-Arnold Networks: a Critique
https://medium.com/@rubenszimbres/kolmogorov-arnold-networks-a-critique-2b37fea2112e

- Reddit Entry about KANs
https://www.reddit.com/r/slatestarcodex/comments/1ciegqt/kolmogorovarnold_networks_paper/?rdt=61654

- A Comprehensive Survey on Kolmogorov Arnold Networks (KAN)
https://arxiv.org/html/2407.11075v5

- ICLR OpenReview
https://openreview.net/forum?id=Ozo7qJ5vZi


# Results of depth_dataset_init_extra_06

xavier_in results in more activation functions used instead of only individual ones:
http://127.0.0.1:5040/#/experiments/262300484374075982/runs/59111317b85e4580bdf530fc65956714/artifacts
like compared to default:
http://127.0.0.1:5040/#/experiments/262300484374075982/runs/340d06beea5948b6a3fc7a8aa05c6b2a/artifacts

# Results of depth_dataset_init_extra_09
When a layer l exceeds the grid (-1,+1) it will be a 0 in the input for layer l+1
Should the first layer have a different (finer) knot vector than the later layers?

default initialization can only train until layer 6.
Training beyond that results in 0.5 accuracy.
The activations in 10 layer moon show that the mean of the last layer gets up, setting all predictions to 1
Default init: the 

# Results of width_dataset_init_10

Default initialization shows functions, that somewhat resemple basis functions.
http://127.0.0.1:5030/#/experiments/901457137512269164/runs/2e3940d127df4f9fac74d17dc6e42ab9/artifacts
Xavier_in_out initialization shows functions that are much more complex.
http://127.0.0.1:5030/#/experiments/901457137512269164/runs/66475a93c1054a58a6ff97bbfab95c1e/artifacts

Default worked good train acc 0.83 and xavier_in_out worked bad 0.57 (but shows an interesting pattern)

Observation
xavier_in_out shows complex b-splines. 
The acts are big [-1.5, 1.5] at initialization already
The coefs are crazy big [-4, 4] at initialization already

xavier_in_out needs to create less variance in layer that are long 5, but thin 1.
restructure formula

Try:
hold the amount of control points (grid_size) small (3), so it is more likely to build a basis function.
Maybe by this it can create better symbolic regressions than default? 


Initialize with normal distribution so it occurs less that the grid borders are reached (-1,1).


# Results of width_dataset_init_14

This run experiments with a low grid size of 1, hidden_depth = 5, hidden_width of [1,5] and datasets [random, moon].
Looking at the kan_actiations-violins-extended-initialized.png we compare two different initialization modes.
default and xavier_in_out.
default: http://127.0.0.1:5030/#/experiments/878956549760550289/runs/e2a5c17b575345eba2e4d42dc104db3a/artifacts
xavier_in_out: http://127.0.0.1:5030/#/experiments/878956549760550289/runs/e3921c123bba478caa8169a7ec145f97/artifacts

the range of default is between 0.1 and -0.1.
layer 0 has the biggest variance (also two splines instead of one, for the layer 1 ... 5)
the variance at layer 1 is small but notable.
layer 2 ... 5 are collapsed in variance [random garbage because train acc=0.52. the collapsed layers might explain this].

the range of xavier_in_out between 0.5 and -1.0.
the variances apear much more similar to each other.
layer 0 ... 3 have notable variance.
layer 4, 5 are collapsed [this goes against the random garbage theory, because here the train accuracy is 0.83].

The information changes through the network go further with xavier_in_out than for default.
interestingly, in the collapsed layers there is still positional change (does this step from random sampling?)

To test this we need to test only sampling the first 100 always, but that might be hard because of the plotting function

the plotting function for default does not show any activations. 
turn up beta in code base [width_dataset_init_15 tests that].
xavier_in_out shows splines.

The results suggest that xavier_in_out works better than default.
hidden_depth = 1, dataset=moon, grid_size=1.

native_noise reports 0.5,0.5 in mlflow, but 0.58 train accuracy in the plots.
[investigate why there is a difference in this case]

for moon 
these worked. train acc ~0.8:
'width_in' 'xavier_in' 'xavier_torch' 'width_in_num' 'xavier_in_num' 'width_in_out' 'xavier_in_out' 'width_in_out_num' 
these didnt work ~0.5:
'default' 'native_noise' 

Cool, all the methods work better than default and the mostly randomly chosen native_noise.
Which method does it best?
decide that by inspecting their performance when scaling up the model with and depth.

in these results we also have the run where the hidden_width is 5 resulting in a 5,5 architecture.
all models work here.
indicating that increasing the complexity of the model makes the correct initialization less interesting/necessary. 

for the random dataset:
hidden architecture 5,1 [height, width]
all are around 0.5 train_acc.

hidden architecture 5,5 [height, width]
show different results.
default is around 0.5 train_acc.
the others have varying results between 0.519 ... 0.581

-----------------------
Do a grid search and create matrix plots for train_acc on random.
get the ranges from the current experiments 
when does training end and overfitting start?
-----------------------

xavier_in_out has a train_acc of 0.58 on random
http://127.0.0.1:5030/#/experiments/878956549760550289/runs/cb1635243d3044dba2aa1921911fcd2c/artifacts
the coef trained show a decrease until the last layer.
the activations appear to be crazy high -3, 3 also decreasing though the layers.
at initialization the last layers 5 mean is much higher than the others.
layer 3, 4 appear to be a bit more positive as well.
at initialization the range is between 0.4, -0.3.
after training the range is much bigger 3,-3. ten times fold.
the gradoients are in range 0.0003, -0.0002.
[i need a mean value label/title for the violin plots]
the violins_extended plots show 4 seperate categories in the last layer.
There apears to be generally more category building through the layers.
from no category to many to 4 (4 individual activation functions?).
why does this happen at initializaton?
after training there might be categories in the last layer
at initialization most of the activation functions look linear.
Only in the earlier layers are somewhat more complex splines like x^2.
after training the splines are more complex (grid=1) through all layers.
the most complex apear to be in early middle layers 1,2,3

default on random has 0.5 train_acc

why are the activations of xavier_in_out so big and the splines show no sign of being at the border

# Introduction

grid -1, 1
After training a KAN the coef of the b-splines, the mean value of the activations roughtly center around 0.
This indicates, that the coefs closer to 0 are more important then the coefs at the border of the grid -1,1.



# Ideas
- Plot matrix that shows the var coef/activation with width and height on the axes

