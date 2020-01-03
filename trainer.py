#Copy Right @Maxwell Jianzhou Wang
#ELG5131 Graphical Models Project 2
#Classifier Trainer


import numpy as np
import scipy.io
from svm import SVMTrainer, Kernel

def trainer(d, k, saved, gamma):
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

    training_vectors = np.array(class_1_vectors[0:800]+class_2_vectors[0:800])
    training_labels = np.array(class_1_labels[0:800]+class_2_labels[0:800], dtype=float)

    trainer = SVMTrainer(Kernel.rbf(gamma=gamma), 0.1)
    predictor = trainer.train(training_vectors, training_labels)
    np.savetxt('./out/bias_d{}_k{}_saved{}.out'.format(d, k, saved), np.array([predictor._bias]))
    np.savetxt('./out/weights_d{}_k{}_saved{}.out'.format(d, k, saved), predictor._weights)
    np.savetxt('./out/support_vectors_d{}_k{}_saved{}.out'.format(d, k, saved), predictor._support_vectors)
    np.savetxt('./out/support_vector_labels_d{}_k{}_saved{}.out'.format(d, k, saved), predictor._support_vector_labels)

