#!/usr/bin/env python
"""
train_model.py

This script demonstrates training an LSTM network on the preprocessed
Area2_Bump data. It uses a mm‐PHATE TraceHistory callback to record
the LSTM hidden activations during training and then applies mm‐PHATE
to reduce their dimensionality. The example includes a simple 2D visualization.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, History
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

import m_phate
import m_phate.train
import scprep
from scipy.signal import find_peaks


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ["PYTHONHASHSEED"] = str(seed)


class TestCallback(Callback):
    """
    Custom callback to evaluate the model on test data after each epoch.
    The test loss and accuracy are stored in self.history.
    """

    def __init__(self, test_data):
        super(TestCallback, self).__init__()
        self.test_data = test_data
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        logs['test_accuracy'] = acc
        logs['val_loss'] = loss
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)
        self.model.history = self


def main():
    set_seed(42)

    # Load preprocessed data (expects files in the Area2_Bump folder)
    trainX = np.load('Area2_Bump/trainX.npy')
    trainy = np.load('Area2_Bump/trainy.npy')
    testX = np.load('Area2_Bump/testX.npy')
    testy = np.load('Area2_Bump/testy.npy')

    trainy = to_categorical(trainy)
    testy = to_categorical(testy)

    # Configure GPU (if available)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            print(f"{len(gpus)} GPU(s) available.")
        except RuntimeError as e:
            print(e)

    # Select trace samples (few samples per class) for tracking hidden activations
    num_sample = 5
    num_class = trainy.shape[1]
    trace_idx = []
    for i in range(num_class):
        indices = np.argwhere(trainy[:, i] == 1).flatten()
        trace_idx.append(np.random.choice(indices, min(len(indices), num_sample), replace=False))
    trace_idx = np.concatenate(trace_idx)
    x_trace = trainX[trace_idx]

    # Build the LSTM model using the Keras functional API
    num_unit = 20
    dropout_flag = False
    inputs = Input(shape=(trainX.shape[1], trainX.shape[2]))
    lstm_out = LSTM(num_unit, return_sequences=True)(inputs)
    x = Dropout(0.8)(lstm_out) if dropout_flag else lstm_out
    flatten = Flatten()(x)
    outputs = Dense(num_class, activation='softmax')(flatten)
    model = Model(inputs=inputs, outputs=outputs)

    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Prepare callbacks: test evaluation, trace history for mm‐PHATE, and standard history.
    # test_callback = TestCallback((testX, testy))
    trace_callback = m_phate.train.TraceHistory(x_trace, Model(inputs=inputs, outputs=lstm_out))
    history_cb = History()

    # Train the model
    epochs = 200
    batch_size = 64
    verbose = 2
    history = model.fit(trainX, trainy,
                        validation_data=(testX, testy),
                        callbacks=[trace_callback, history_cb],
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbose)

    loss, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    print("Final test accuracy: {:.3f}".format(accuracy))

    # --- mm‐PHATE Analysis ---
    # Downsample epochs and intrinsic steps for visualization.
    intrinsic_steps = 600
    intrinsic_step_sample_size = 100
    epoch_sample_step_after_epoch_30 = 10
    epoch_samples = np.concatenate([np.arange(29), np.arange(29, epochs, epoch_sample_step_after_epoch_30)])
    intrinsic_step_samples = np.linspace(0, intrinsic_steps - 1, intrinsic_step_sample_size, dtype=int)

    # Reshape and normalize trace data for mm‐PHATE
    trace_data = np.array(trace_callback.trace).transpose(0, 2, 1, 3)[epoch_samples]
    trace_data = trace_data[:, intrinsic_step_samples, :, :]
    trace_data_norm = m_phate.utils.normalize(trace_data)
    trace_data_reshaped = trace_data.reshape(len(epoch_samples) * intrinsic_step_sample_size, num_unit,
                                             num_sample * num_class)

    m_phate_op = m_phate.M_PHATE(n_jobs=1)
    mphate_data = m_phate_op.fit_transform(trace_data_reshaped)
    print("mm‐PHATE output shape:", mphate_data.shape)

    # Example 2D visualization using scprep (colors by intrinsic time step)
    mphate_2D = True
    if mphate_2D:
        # Labels for coloring
        intrinsic_steps_plot = np.tile(intrinsic_step_samples, len(epoch_samples))
        intrinsic_steps_plot = np.repeat(intrinsic_steps_plot, num_unit) # timestep index for each node
        epoch_label = np.repeat(epoch_samples, intrinsic_step_sample_size * num_unit) # epoch index for each node
        # hidden unit index for each node
        unit = np.tile(np.arange(trace_data_reshaped.shape[1]), len(epoch_samples) * intrinsic_step_sample_size)

        # the label of each digit we selected
        digit_ids = np.repeat(np.arange(num_class), num_sample)
        digit_activity = np.empty((len(epoch_samples), num_class, intrinsic_step_sample_size, num_unit))
        for idx, the_epoch in enumerate(epoch_samples):
            # the average activity over digit labels for each element of the flattened trace
            trace_data_norm_sample = trace_data_norm[idx]
            digit_activity[idx] = np.array(
                [np.sum(np.abs(trace_data_norm_sample[:, :, digit_ids == digit]), axis=2)
                 for digit in np.unique(digit_ids)])
        # the digit label with the highest average activity for each element of the flattened trace
        most_active_output = np.empty((len(epoch_samples), intrinsic_step_sample_size * num_unit))
        for idx, the_epoch in enumerate(epoch_samples):
            most_active_output[idx] = np.argmax(digit_activity[idx], axis=0).flatten()
        most_active_output = most_active_output.flatten()

        scprep.plot.scatter2d(mphate_data, c=epoch_label,
                              label_prefix="mm‐PHATE",
                              legend_title="Epoch",
                              figsize=(8, 8))
        plt.show()

        scprep.plot.scatter2d(mphate_data, c=intrinsic_steps_plot,
                              label_prefix="mm‐PHATE",
                              legend_title="Timestep",
                              figsize=(8, 8))
        plt.show()

        scprep.plot.scatter2d(mphate_data, c=unit,
                              label_prefix="mm‐PHATE",
                              legend_title="Hidden Unit",
                              figsize=(8, 8))
        plt.show()

        scprep.plot.scatter2d(mphate_data, c=most_active_output,
                              label_prefix="mm‐PHATE",
                              legend_title="Most Active Output",
                              figsize=(8, 8))
        plt.show()


if __name__ == '__main__':
    main()
