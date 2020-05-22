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