
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


14.09.2025
Spline Activations
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/d7ace100ca844b6d98c6effa597a0bda/artifacts
The last layer creates super high activations, due to Sigmoid being added by the CrossEntropyLoss
The networks tries to create very high output activations.
this leads to the problem, that especially at the second to last layer a lot of activations are outside the spline grid and thus the spline does not add meaningful information
is this one of the reasons, why deep KANs (depth: 20) are not possible, or why the accuracy drops of?
Does Batch Normalization fix this issue and allow for deeper KANs?

spiral preliminary capacity test
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/640df449ce204163897fb61f5bf0f15f/artifacts
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/69f945b0c30547bc8aa3ec8e3bc828b5/artifacts
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/fc819d92ebf245c4a046e27c6efcac43/artifacts
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/6fbe91f04d794767937331d3f69bd602/artifacts
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/d7ace100ca844b6d98c6effa597a0bda/artifacts
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/2ddacd4799fb4bd2a3f5fc1cfd1908db/artifacts
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/a9e8fed407bc4e6c882b3f4b65e0597c/artifacts
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/0c9ed4955bbf452fa3cf84056506dabe/artifacts


edge_forward_spline_n
1e-6
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/f600d2e2ba964222b79630b861e04805/artifacts
1e-5
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/207fa6ad325444bcb86fe59a02c0e11e/artifacts
1e-4
http://127.0.0.1:9099/#/experiments/987478017712724477/runs/ba0ce07478dd4be3b30b499fbdb0b7b7/artifacts

Cross Entropy vs. MSE
depth_capacity_boxes_19


MLFLOW ERROR
  File "/home/student/l/luniehaus/cv_home/dev-uos/miniconda/envs/pykan/lib/python3.9/contextlib.py", line 126, in __exit__                                                                                                                                                                                                                                                                                                      [25/1940]
    next(self.gen)                                                                                                                                                                                                                                                                                                                                                                                                                       
  File "/home/student/l/luniehaus/cv_home/dev-uos/miniconda/envs/pykan/lib/python3.9/site-packages/mlflow/tracking/client.py", line 2070, in _log_artifact_helper                                                                                                                                                                                                                                                                        
    self.log_artifact(run_id, tmp_path, artifact_dir)                                                                                                                                                                                                                                                                                                                                                                                    
  File "/home/student/l/luniehaus/cv_home/dev-uos/miniconda/envs/pykan/lib/python3.9/site-packages/mlflow/tracking/client.py", line 2000, in log_artifact                                                                                                                                                                                                                                                                                
    self._tracking_client.log_artifact(run_id, local_path, artifact_path)                                                                                                                                                                                                                                                                                                                                                                
  File "/home/student/l/luniehaus/cv_home/dev-uos/miniconda/envs/pykan/lib/python3.9/site-packages/mlflow/tracking/_tracking_service/client.py", line 918, in log_artifact                                                                                                                                                                                                                                                               
    artifact_repo = self._get_artifact_repo(run_id)                                                                                                                                                                                                                                                                                                                                                                                      
  File "/home/student/l/luniehaus/cv_home/dev-uos/miniconda/envs/pykan/lib/python3.9/site-packages/mlflow/tracking/_tracking_service/client.py", line 901, in _get_artifact_repo                                                                                                                                                                                                                                                         
    artifact_repo = get_artifact_repository(artifact_uri)                                                                                                                                                                                                                                                                                                                                                                                
  File "/home/student/l/luniehaus/cv_home/dev-uos/miniconda/envs/pykan/lib/python3.9/site-packages/mlflow/store/artifact/artifact_repository_registry.py", line 131, in get_artifact_repository                                                                                                                                                                                                                                          
    return _artifact_repository_registry.get_artifact_repository(artifact_uri)                                                                                                                                                                                                                                                                                                                                                           
  File "/home/student/l/luniehaus/cv_home/dev-uos/miniconda/envs/pykan/lib/python3.9/site-packages/mlflow/store/artifact/artifact_repository_registry.py", line 76, in get_artifact_repository                                                                                                                                                                                                                                           
    return repository(artifact_uri)
  File "/home/student/l/luniehaus/cv_home/dev-uos/miniconda/envs/pykan/lib/python3.9/site-packages/mlflow/store/artifact/mlflow_artifacts_repo.py", line 51, in __init__
    super().__init__(self.resolve_uri(artifact_uri, get_tracking_uri()))
  File "/home/student/l/luniehaus/cv_home/dev-uos/miniconda/envs/pykan/lib/python3.9/site-packages/mlflow/store/artifact/mlflow_artifacts_repo.py", line 65, in resolve_uri
    _validate_uri_scheme(track_parse)
  File "/home/student/l/luniehaus/cv_home/dev-uos/miniconda/envs/pykan/lib/python3.9/site-packages/mlflow/store/artifact/mlflow_artifacts_repo.py", line 35, in _validate_uri_scheme
    raise MlflowException(
mlflow.exceptions.MlflowException: When an mlflow-artifacts URI was supplied, the tracking URI must be a valid http or https URI, but it was currently set to file:///net/store/cv/users/luniehaus/projects/pykan_experiments/pykan/tutorials/Experiments/trainer/mlruns. Perhaps you forgot to set the tracking URI to the running MLflow server. To set the tracking URI, use either of the following methods:
1. Set the MLFLOW_TRACKING_URI environment variable to the desired tracking URI. `export MLFLOW_TRACKING_URI=http://localhost:5000`
2. Set the tracking URI programmatically by calling `mlflow.set_tracking_uri`. `mlflow.set_tracking_uri('http://localhost:5000')`
