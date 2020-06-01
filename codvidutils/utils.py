__author__ = '@iRodGit'


def weigths2loss(Y):
    """This function calculates the weighted for imbalanced train set.
    It returns a dictionary with the labels as keys and the weight are..."""
    from numpy import unique
    if len(unique(Y)) != 2:
        raise ValueError("Y must have only two labels 0 and 1.")
    else:
        num_0_class = (Y[Y== 0]).shape[0]
        num_1_class = (Y[Y== 1]).shape[0]
        
        dict_weigths = {0: num_1_class/ (num_0_class+num_1_class), 1: num_0_class/ (num_0_class+num_1_class)}
        
    return dict_weigths

def adding_images(X, Y,strides= 5):#kernel =None ):
    from codvidutils import nwpic as nw
    from numpy import concatenate, ones
    new_X = nw.new_pictures_arrays(X[Y==1],strides)
    X = X[:,10:190,10:190]
    new_Y = ones(new_X.shape[0])
    X = concatenate([X,new_X],axis=0)
    Y = concatenate([Y,new_Y],axis=0)
    return X, Y

def load_pictures (data_frame,data_dir,channels):
    import numpy as np
    from PIL import Image
    pics = []
    for img in data_frame['image_path'].values:
        if channels ==3:
            pics.append(np.array(Image.open(data_dir + img))[:, :,:3])
        else:
            pics.append(np.array(Image.open(data_dir + img))[:, :,0])

    return np.array(pics)

def get_counts(Pred, Upre, true_Y,nbins, label):
    import numpy as np
    from scipy.stats import norm
    mean = Pred[true_Y == label]
    unce = Upre[true_Y == label]
    bins = np.linspace(0,1,nbins+1)
    probability = np.zeros([mean.shape[0], nbins])
    print(probability.shape)

    for i in range(mean.shape[0]):
        for j in range(nbins):
            probability[i,j] = (norm.cdf(bins[j+1],loc=mean[i],scale=unce[i])-norm.cdf(bins[j],loc=mean[i],scale=unce[i]))
    
    counts = np.sum(probability,axis = 0)
    variance = np.multiply(probability,(1- probability))
    variance = np.sum(variance,axis=0)
    uncer_ counts = np.sqrt(variance)
    return counts, uncer_counts

def get_counts_by_class(Pred,Upre, true_Y, nbins):
    from numpy import unique, linspace
    dic_counts = {0: [], 1:[],2:[] }
    dic_uncounts = {0: [], 1:[],2:[] }
    for label in unique(true_Y):
        print(label)
        dic_counts[label], dic_uncounts[label] = get_counts(Pred,Upre,true_Y,nbins, label)

    bins = linspace(0,1,nbins+1)
    return dic_counts, dic_uncounts, bins

def hist_with_unce (hist_bins, counts, uncer_counts, color1, color2,lw=2, label = None, histtype = "step"):
    import matplotlib.pyplot as plt
    """
    hist_bins: array with the bin edges, (N+1)-array
    counts: N-array with the value of each bin
    uncer_counts: N-array with the uncertainty of each count.
    """
    plt.hist(hist_bins[:-1], hist_bins, weights = counts,histtype=histtype,color=color1,lw=lw,label =label)
    for i in range(counts.shape[0]):
        plt.fill_between(hist_bins[i:i+2], counts[i]-uncer_counts[i],counts[i]+uncer_counts[i], color=color2)

    