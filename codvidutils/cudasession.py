__author__ = '@Tssp'


def set_session(ngpu=None, ncpu=None):
    ''' This function sets the number of devices
    you are going to work with.
    Inputs: ngpu: number of GPUs.
            ncpu: number of CPUs.
            mode: "max" (default)
    Examples:
    - max:
        -> set_session()
    - And the manual mode is used if number of gpus and cpus is specified:
        -> set_session(1, 3)
    '''
    import tensorflow.compat.v1 as tf
    from keras import backend as K
    import multiprocessing
    gpus = 2
    cpus = multiprocessing.cpu_count()
    print("Num GPUs Available: ", gpus)
    print("Num CPUs Available: ", cpus)
    if (ngpu is None) and (ncpu is None):
        config = tf.ConfigProto(device_count={'GPU': gpus, 'CPU': cpus})
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
        print('---------Keras session created with---------\n - {} GPUs\n - {} CPUs'.format(gpus, cpus))
    elif (ngpu is not None) and (ncpu is not None):
        config = tf.ConfigProto(device_count={'GPU': ngpu, 'CPU': ncpu})
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
        print('---------Keras session created with---------\n - {} GPUs\n - {} CPUs'.format(ngpu, ncpu))
    else:
        raise ValueError('There are only two modes: manual and max.')