from inspect import currentframe, getframeinfo
from datetime import datetime
import numpy as np
import sys


# ----------------------------------------------------------------------------------------------------------------------
#                                               	Model Plot
# ----------------------------------------------------------------------------------------------------------------------

def plot_model_history(model_history):
	# Might not be needed - Pytorch and Keras have similar functionality
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
#                                               	Probability Sample
# ----------------------------------------------------------------------------------------------------------------------

def random_sample(diversity=0.5):
    def random_samp(preds, argmax=True):
        """Helper function to sample an index from a probability array"""
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / diversity

        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        # Randomly draw smaples from the multinomial distrubtion with 101 probability classes
        if argmax:
            return np.argmax(probas)
        else:
            return probas

    return random_samp


def max_sample():
    def m(preds):
        return np.argmax(preds)

    return m
