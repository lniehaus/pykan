
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

