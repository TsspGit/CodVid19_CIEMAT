__author__ = '@iRodGit'



def plot_histogram(y_true, y_pred,labels):
    
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))#,6
    plt.style.use('seaborn-deep') 
    plt.grid(True)

    lw=2
    for value in np.unique(y_true):
        plt.hist(y_pred[(y_true==value)],50,histtype='step',lw=lw, label= labels[value])
    
    plt.ylabel('Number of events /0.01')
    plt.xlabel('Score')
    plt.yscale("log")
    plt.legend( loc='upper center')
    plt.show()
        
    return 

def plot_metric(history, name):
    
    if name == "mse":
        name = "mean_squared_error"
        
    ylabel_dic = {"mean_squared_error":"Mean Squared Error as metric" , "loss": "Binary Cross-Entropy as loss", "acc": "Accuracy as metric"}
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))#,6
    plt.style.use('seaborn-deep') 
    plt.grid(True)
    
    plt.plot(history.history[name], color="blue", label="Training data set")
    plt.plot(history.history["val_"+name], color="orange", label="Testing data set")
    plt.ylabel(ylabel_dic[name])
    plt.xlabel('Epochs')
    plt.legend(loc='upper center')
    plt.show()
        
    return 