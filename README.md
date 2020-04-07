# CodVid19 CIEMAT

## Authors: 
Miguel Cárdenas, Iñaki Rodríguez, Tomás Sánchez working at the Computation Unity of CIEMAT

### Outline
* [Getting started](#1-getting-started)
* [Data and Information](#2-data-and-information)
* [Folders explanation](#3-folders-explanation)

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
