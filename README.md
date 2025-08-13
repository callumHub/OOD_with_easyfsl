# ATTENTION
In order to run most easyFSL with an MLP backbone/encoder, modifying the libraries assert statements and methods was required. The modified easyFSL is put in the directory modified_easy_fsl. Replace .venv/Lib/site_packages/easyfsl with this one to have model_runner.py run without errors. Note, you will need to create a new parameter_store in parameter_store.py if your dataset has a different number of input features than 128.  

It is possible to extend each broken method to avoid modifying the library directly. The library modification was quick and dirty and done for the sake of getting research results. 

Therefore, I have also pushed the easyFSL library with my code, that includes the modifications I have made. 

## Using easyFSL
on their github: [easyfsl example notebook](https://github.com/sicara/easy-few-shot-learning/blob/master/notebooks/episodic_training.ipynb)
Adapt the code in this notebook to your use case: 
For a custom dataset, make sure it is in the structure ["data", "labels"] and load it with easyFSL load_feature_dataset function. 
Then define your own custom backbone with pytorch. In the cell where the initialize PrototypicalNetworks, replace convolutional network with your custom model. Then the library may throw errors which you can either dirty modify the library or extend the method throwing the errors to work with you new backbone. 

Some library modifications may require reshaping the mlp output with a redundant dimension to pass some checks. 


##  Using this repo
See model_runner.py for a basic run example, relies on methods in 'prototype_model_runner.py' Learning and using these methods is likely more difficult than the above method, as this code was built rapidly and thus is ridgid and hard to extend
