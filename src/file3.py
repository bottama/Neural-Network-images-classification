""" Accuracy Comparison """

# import modules
import utils
from tensorflow.keras import utils as tf_utils
import numpy as np
from scipy.stats import norm, ttest_rel
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load the test CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = utils.load_cifar10()

    # Pre-processing
    # Normalize each pixel of each channel so that the range is [0, 1];
    # each pixel is represented by an integer value in the 0-255 range.
    x_train, x_test = x_train / 255., x_test / 255.

    # Create one-hot encoding of the labels;
    # pre-process targets in order to perform multi-class classification.
    n_classes = 3
    y_train = tf_utils.to_categorical(y_train, n_classes)
    y_test = tf_utils.to_categorical(y_test, n_classes)

    # Load the trained models
    model_task2 = utils.load_keras_model('../deliverable/nn_task2.h5')
    model_task1 = utils.load_keras_model('../deliverable/nn_task1.h5')

    # Predict on the given samples
    y_pred_task1 = model_task1.predict(x_test)
    y_pred_task2 = model_task2.predict(x_test)

    # prepare data for f1 score:
    y_test_f1 = np.argmax(y_test, axis=1).astype(int)
    y_pred_task1_f1 = np.argmax(y_pred_task1, axis=1).astype(int)
    y_pred_task2_f1 = np.argmax(y_pred_task2, axis=1).astype(int)

    """ Accuracy comparison between model T1 and model T2 """

    # data check
    assert y_test.shape == y_pred_task1.shape
    assert y_test.shape == y_pred_task2.shape

    # correct classifications
    e_task1 = (np.argmax(y_pred_task1, axis=1) == np.argmax(y_test, axis=1)).astype(int)
    e_task2 = (np.argmax(y_pred_task2, axis=1) == np.argmax(y_test, axis=1)).astype(int)

    # ratio of correct classifications
    mean_e_task1 = e_task1.mean()
    mean_e_task2 = e_task2.mean()

    # sample variance
    s2_task1 = mean_e_task1 * (1 - mean_e_task1)
    s2_task2 = mean_e_task2 * (1 - mean_e_task2)

    # results -- mean -- s2
    print('model T1: mean: {} -- s2: {}'.format(mean_e_task1, s2_task1))
    print('model T2: mean: {} -- s2: {}'.format(mean_e_task2, s2_task2))

    # Test statistics
    l = len(x_test)
    T = (mean_e_task2 - mean_e_task1)
    T /= np.sqrt(s2_task2 / l + s2_task1 / l)

    # results -- T
    if abs(T) > norm.ppf(.975):
        print('T={} is not in 95% confidence interval (-1.96, 1.96)'.format(T))
    else:
        print('T={} is in 95% confidence interval (-1.96, 1.96)'.format(T))

    # t-test and p-value
    tt, p_val = ttest_rel(e_task2, e_task1)
    print('t-test: T={:.2f}, p-value={:.4f}'.format(tt, p_val))

    # Assessing anomaly detection methods

    # assess the highest values
    y_pred_task1 = model_task1.predict(x_test).ravel()
    y_pred_task2 = model_task2.predict(x_test).ravel()
    y_test = y_test.ravel()

    # FPR -- TPR -- THRESHOLDS
    # FPR = false positive rate
    # TPR = true positive rate

    fpr_task1, tpr_task1, thresholds_task1 = roc_curve(y_test, y_pred_task1)
    fpr_task2, tpr_task2, thresholds_task2 = roc_curve(y_test, y_pred_task2)

    # Area Under Curve: AUC
    auc_task1 = auc(fpr_task1, tpr_task1)
    auc_task2 = auc(fpr_task2, tpr_task2)
    print('AUC: model task 1 ={:.4f} -- model task 2 ={:.4f}'.format(auc_task1, auc_task2))

    # Receiver Operating Characteristic Curve -- ROC Curve
    def roc_curve():
        plt.plot(fpr_task1, tpr_task1, label='AUC model task 1: {:.4f}'.format(auc_task1))
        plt.plot(fpr_task2, tpr_task2, label='AUC model task 2: {:.4f}'.format(auc_task2))
        plt.title('Receiver Operating Characteristic Curve [ROC Curve]')
        plt.xlabel('False Positive Rate [ FPR ]')
        plt.ylabel('True Positive Rate [ TPR ]')
        plt.plot([0, 1], [0, 1], linestyle='--', dashes=(3, 1), color='black')
        plt.legend()
        return plt.show()


    # F1 score
    f1_task1 = f1_score(y_true=y_test_f1, y_pred=y_pred_task1_f1, average='micro')
    f1_task2 = f1_score(y_true=y_test_f1, y_pred=y_pred_task2_f1, average='micro')
    print('F1 score: model task 1 ={:.4f} -- model task 2 ={:.4f}'.format(f1_task1, f1_task2))

    # roc curve plot
    roc_curve()
