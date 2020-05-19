# CodVid19 CIEMAT

## Authors: 
Miguel Cárdenas, Iñaki Rodríguez, Tomás Sánchez working at the Computation Unity of CIEMAT

### Outline
* [Getting started](#1-getting-started)
* [Data and Information](#2-data-and-information)
* [Folders explanation](#3-folders-explanation)
* [Main codes](#4-main-codes)

## 1. Getting started
Firstly, you have to add an ssh key to your github account. To do so, just execute on the terminal:
```
$ ssh-keygen
```
And press enter to everything.

Then, copy the content on .ssh/id_rsa.pub into your ssh keys on GitHub.

Once you did that you only need to run:
```
$ git clone git@github.com:TsspGit/CodVid19_CIEMAT.git
```
And the project would be copied to your current folder.

## 2. Data and Information
The info about the data and how to download it is in:
https://github.com/lindawangg/COVID-Net

Link to download the images:
https://drive.google.com/open?id=1AQDqGTAn4XshuR76LNmHnK70TkFpAzRq

## 3. Folders explanation
### data/
In this folder you can find a **resize/** folder with a python code for input the images and resize them. Also you can find a train and a test folder with the corresponding images.

### Figures/
Autodescriptive folder name.

### codvidutils/
This folder serves for being imported as a python module. 
- **cudasession.py**: creates de Keras session with the wanted number of CPUs and GPUs.
- **imageproc.py**: contains a function to create the categorical variables in the dataset.
- **nwpic.py**: umbalancing and oversampling methods.
- **plotutils.py**: histograms.
- **Transformation_class.py**: this is the body of our analysis, it contains every function needed to preprocess the dataset until it feeds the neural network architectures.
- **Autoencoder_Transformation_main.py**: applies the last class and feed the best Autoencoder model with the outputs. Once is done a new dataset is created with the code of the original images.

### hdf_files/
Saved neural network trained models.

### log/
Outputs from Random Forests and XGBoosts algorithms. That means: the logs of the executions and the predictions.

## 4. Main codes:
### Covid_autoencoder.ipynb:
Uses Transformation_class.py to prepare the dataset and train the autoencoder. The best autoencoder model is saved thanks to the Keras Callbacks. Finally, it can be seen the reconstruction of the original images by the autoencoder, as well as the code that it's going to be used as input for the classification algorithms.

### Random_Forest_Regressor.py:
Uses Autoencoder_Transformation_main.py to obtain the codified train and test images and classify them with Random Forests, the outputs are saved in **log/preds_RFr_v4.txt**.

### XGBoost_Regressor.py:
Same than Random_Forest_Regressor.py. The predictions are saved in **log/preds_XGBr_lr{}_ n{} _maxdepth{}.txt**, depends on the parameters used. The best configuration is:
- lr = 0.005
- n = 250
- maxdepth = 3

### Random_Forest_Classification.ipynb:
Analyzes the results with a confussion matrix and an histogram depending on the score obtained.

### XGBoost_Classification.ipynb:
Same vibes than Random_Forest_Classification.ipynb, this time analyzing XGBoost results.

### Classical_CNN.ipynb:
(Iñaki)

### Inception_uncertainty.ipynb:
(Iñaki)

### Inception_uncertainty_TF.ipynb:
(Iñaki)

## Transfer Learning.ipynb:
(Iñaki)
