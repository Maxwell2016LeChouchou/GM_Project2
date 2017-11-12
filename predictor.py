import numpy as np
import numpy.linalg as la
import scipy.io
from svm import SVMTrainer, Kernel

def predictor(d, k, saved, gamma):
    # input data file
    mat = scipy.io.loadmat('observed/classify_d{}_k{}_saved{}.mat'.format(d, k, saved))

    # construct vectors
    class_1_vectors = []
    class_1_labels = []
    class_2_vectors = []
    class_2_labels = []
    count = 0
    while (count < 1000):
        vector3 = []
        for index in range(0, d):
            vector3.append(mat.values()[0][index][count])
        class_1_vectors.append(vector3)
        class_1_labels.append(1.0)
        vector3 = []
        for index in range(0, d):
            vector3.append(mat.values()[2][index][count])
        class_2_vectors.append(vector3)
        class_2_labels.append(-1.0)
        count = count + 1

    testing_vectors = np.array(class_1_vectors[800:1000]+class_2_vectors[800:1000])

    bias = np.loadtxt('./out/bias_d{}_k{}_saved{}.out'.format(d, k, saved))
    weights = np.loadtxt('./out/weights_d{}_k{}_saved{}.out'.format(d, k, saved))
    support_vectors = np.loadtxt('./out/support_vectors_d{}_k{}_saved{}.out'.format(d, k, saved))
    support_vector_labels = np.loadtxt('./out/support_vector_labels_d{}_k{}_saved{}.out'.format(d, k, saved))

    r = [0] * 400
    for j, x_j in enumerate(testing_vectors):
        result = 0.0
        for z_i, x_i, y_i in zip(weights,
                                 support_vectors,
                                 support_vector_labels):
            result += z_i * y_i * np.exp(-gamma*la.norm(np.subtract(x_i, x_j)))
        r[j] = result
    r = np.sign(r)
    print (r[0:200])
    print (r[200:400])

    error_count = 0
    for index in range(0, 200):
        if r[index] != 1.0:
            error_count += 1
    for index in range(200, 400):
        if r[index] != -1.0:
            error_count += 1

    print('Error Count: {}'.format(error_count))
