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