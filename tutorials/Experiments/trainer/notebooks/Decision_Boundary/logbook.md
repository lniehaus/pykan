
Decision_Boundary_Analysis_4: Currently the Weighted Layer Training does not work
I am creating Decision_Boundary_Analysis_5 to check this bug

How to reduce the amount of Connected Regions (Islands)?
- Initialize Linearly
- Weighted Layer Training, with weighted gradient step?



next experiment
---------------
base_fun: identity
10 layers deep



create a kolmogorov arnold network, where the coefficients are monotonic increasing for each spline. each spline is using the de boor algorithm and has a grid of 8 and a degree of 3. train it with the data from dataset. create a gif, where each frame shows the decision boundary per epoch. use tqdm for progress and print the loss for train and test and accuracy for train and test. plot each spline of the kolmogorov arnold network before and after training.


create a kolmogorov arnold network, where the coefficients are monotonic increasing for each spline. each spline is using the de boor algorithm and has a grid of 8 and a degree of 3. create and execute a function, which plots each spline of the kolmogorov arnold network

in the KAN class the mode parameter means monotonic mode. mode="default" means no monotonic enforcement. mode="abs" means monotonic mode enabled. create two experiments in Decision_Boumdary_Analysis_8.ipynb with the dataset provided there. use the KAN class like in trainer. one experiment is default and the other experiment enforces monotonic splines with mode="abs". plot the train and test loss. plot the train and test accuracy. plot the decision boundary and count the amount of linear contiguous regions for both experiments. create a gif for each experiment for the development of the decision boundary, where each frame is one epoch. make the playable in the notebook. use tqdm to show the process of the steps. please make sure that the code is correct, since i will get fired if there are errors in it.


Investigate this error:
It happens sometimes when training.
It probably happpens, when the loss gets nan.
If that is so, why does it go to nan?
Stack Trace:
  File "/net/store/cv/users/luniehaus/projects/pykan_experiments/pykan/tutorials/Experiments/trainer/src/trainer.py", line 545, in <module>                                                                 
    main()                                                                                                                                                                                                  
  File "/net/store/cv/users/luniehaus/projects/pykan_experiments/pykan/tutorials/Experiments/trainer/src/trainer.py", line 328, in main                                                                     
    results = model.fit(dataset,                                                                                                                                                                            
  File "/net/store/cv/users/luniehaus/projects/pykan_experiments/pykan/kan/MultKAN.py", line 1618, in fit                                                                                                   
    self.update_grid(dataset['train_input'][train_id])
  File "/net/store/cv/users/luniehaus/projects/pykan_experiments/pykan/kan/MultKAN.py", line 751, in update_grid
    self.update_grid_from_samples(x)
  File "/net/store/cv/users/luniehaus/projects/pykan_experiments/pykan/kan/MultKAN.py", line 745, in update_grid_from_samples
    self.act_fun[l].update_grid_from_samples(self.acts[l])
  File "/net/store/cv/users/luniehaus/projects/pykan_experiments/pykan/kan/KANLayer.py", line 316, in update_grid_from_samples
    self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)
  File "/net/store/cv/users/luniehaus/projects/pykan_experiments/pykan/kan/spline.py", line 131, in curve2coef
    return coef


Add quantitative numbers for connected regions count, so they can be filtered in mlflow


depth_capacity_boxes_04
The results show that the classes are differentiated on 45 degree angle between x and y axis.
http://127.0.0.1:9090/#/experiments/148942313483143531/runs/5d9e4ddf3cf640f7878731ea5076e353/artifacts
The Splines are also very linear
http://127.0.0.1:9090/#/experiments/148942313483143531/runs/5d9e4ddf3cf640f7878731ea5076e353/artifacts
this could indicate, that more grid points are needed to separate the datapoints sufficiently
More grid points (5->40) enables to show the box pattern
http://127.0.0.1:9090/#/experiments/802285876068926351/runs/6af49a932e704069bbb2fc2b49728573/artifacts
 

| train_loss: 2.80e-02 | test_loss: 1.02e-01 | reg: 0.00e+00 | :  10%| | 10178/100000 [03:32<32:39

