import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

plt.rcParams["font.family"] = "Cambria"
plt.rcParams['font.size'] = 10

dataframe = pd.read_csv('vibr/good/DRG.csv', header=None)
DRG = dataframe.values
(f, Pa) = scipy.signal.welch(DRG, 25000, window='hann', scaling='spectrum', axis=1)
Pa_log = 10 * np.log10(Pa)

data = Pa_log

train_data, test_data = train_test_split(
    data, test_size=0.3, random_state=1
)

row_train, col_train = train_data.shape
row_test, col_test = test_data.shape

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

class FaultDetector(Model):
    def __init__(self):
        super(FaultDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),  # 32
            layers.Dense(32, activation="relu"),  # 16
            layers.Dense(16, activation="relu")])  # 8

        self.decoder = tf.keras.Sequential([
            #layers.Dense(16, activation="relu"),  # nie bylo tej
            layers.Dense(32, activation="relu"),  # 16
            layers.Dense(64, activation="relu"),  # 32
            layers.Dense(129, activation="sigmoid")])  # no of samples

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = FaultDetector()

autoencoder.compile(optimizer='adam', loss='mse')

# Training on normal gearbox

history = autoencoder.fit(train_data, train_data,
                          epochs=15,  # 10
                          batch_size=15,  # 20
                          validation_data=(test_data, test_data),
                          shuffle=True)

autoencoder.encoder.summary()

fig_loss, ax_loss = plt.subplots()
ax_loss.plot(history.history["loss"], 'b', label="Training loss")
ax_loss.plot(history.history["val_loss"], 'r', label="Validation loss")
ax_loss.legend()
fig_loss.set_figheight(2.4)
ax_loss.set_xlabel('Epoch number')
ax_loss.set_ylabel('Mean absolute error')
ax_loss.grid(visible=True)
plt.tight_layout()
plt.show()  # overfitting = train loss is less than validation loss

reconstructions = autoencoder.predict(test_data)
train_loss = tf.keras.losses.mse(reconstructions, test_data)

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

fig_tr, ax_tr = plt.subplots()
ax_tr.hist(train_loss[None, :], bins=30)
fig_tr.set_figheight(2.4)
ax_tr.set_xlabel("Validation mean absolute error (gear pair I)")
ax_tr.set_ylabel("Density")
#plt.title("Training data")
plt.axvline(x=threshold, color='r')
plt.tight_layout()
plt.show()

count_g = plt.hist(train_loss[None, :], bins=[np.min(train_loss[None, :]), threshold, np.max(train_loss[None, :])])
plt.xlabel("Mean absolute error")
plt.ylabel("Density")
plt.title("Training data")
plt.axvline(x=threshold, color='r')
plt.show()

G = count_g[0]
GG = G[0] / sum(G) * 100
GF = G[1] / sum(G) * 100
print('GG:', GG)
print('GF:', GF)

encoded_data = autoencoder.encoder(test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

fig_xr1, axr1 = plt.subplots()
fig_xr1.set_figheight(2.4)
axr1.plot(f, test_data[0], 'b')
axr1.plot(f, decoded_data[0], 'r')
axr1.set_xlabel("Frequency, Hz")
axr1.set_ylabel("x, r")
axr1.fill_between(f, decoded_data[0], test_data[0], color='lightcoral')
axr1.legend(labels=['Input', "Reconstruction", "Error"])
axr1.grid(visible=True)
plt.tight_layout()
plt.show()
fig_xr1.savefig('fig_11.png', dpi=300)

# Testing on faulted gearbox

dataframe2 = pd.read_csv('drgania/drg_fail/DRG_fail.csv', header=None)
DRG2 = dataframe2.values
(f, Pa2) = scipy.signal.welch(DRG2, 25000, window='hann', scaling='spectrum', axis=1)
Pa_log2 = 10 * np.log10(Pa2)

data2 = Pa_log2

min_val = tf.reduce_min(data2)
max_val = tf.reduce_max(data2)

data2 = (data2 - min_val) / (max_val - min_val)
data2 = tf.cast(data2, tf.float32)

row_fail, col_fail = data2.shape
print('fail data size:', data2.shape)

encoded_data = autoencoder.encoder(data2).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

fig_xr2, axr2 = plt.subplots()
fig_xr2.set_figheight(2.4)
axr2.plot(f, data2[0], 'b')
axr2.plot(f, decoded_data[0], 'r')
axr2.set_xlabel("Frequency, Hz")
axr2.set_ylabel("x, r")
axr2.fill_between(f, decoded_data[0], data2[0], color='lightcoral')
axr2.legend(labels=['Input', "Reconstruction", "Error"])
axr2.grid(visible=True)
plt.tight_layout()
plt.show()

reconstructions = autoencoder.predict(data2)
test_loss = tf.keras.losses.mse(reconstructions, data2)

fig_tr2, ax_tr2 = plt.subplots()
fig_tr2.set_figheight(2.4)
ax_tr2.hist(test_loss[None, :], bins=30)
ax_tr2.set_xlabel("Validation mean absolute error (gear pair II)")
ax_tr2.set_ylabel("Density")
plt.axvline(x=threshold, color='r')
plt.tight_layout()
plt.show()

count_f = plt.hist(test_loss[None, :], bins=[np.min(train_loss[None, :]), threshold, np.max(train_loss[None, :])])
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.title("Fail")
plt.axvline(x=threshold, color='r')
plt.show()

# Confusion matrix and scores

F = count_f[0]
FF = F[1] / sum(F) * 100
FG = F[0] / sum(F) * 100
print('FF:', FF)
print('FG:', FG)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.summer):
    fig, ax = plt.subplots()
    mat = ax.matshow(df_confusion, cmap=cmap)  # imshow
    ax.xaxis.set_ticks_position('bottom')
    #ax.set_title(title)
    cbar = fig.colorbar(mat)
    # cbar.ax.set_ylabel('%', rotation=0)
    cbar.ax.yaxis.set_major_formatter('{x:1.0f}%')
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=0)
    plt.yticks(tick_marks, df_confusion.index, rotation=90)
    cf = pd.DataFrame.to_numpy(df_confusion)
    plt.text(0, 0, str(math.floor(cf[0, 0] * 10) / 10) + '%', horizontalalignment='center')  # FF
    plt.text(0, 1, str(math.floor(cf[1, 0] * 10) / 10) + '%', horizontalalignment='center')  # GF
    plt.text(1, 0, str(math.ceil(cf[0, 1] * 10) / 10) + '%', horizontalalignment='center')   # FG
    plt.text(1, 1, str(math.ceil(cf[1, 1] * 10) / 10) + '%', horizontalalignment='center')   # GG
    ax.set_ylabel('Actual state')
    ax.set_xlabel('Predicted state')
    plt.show()

cf = np.array([[FF, FG], [GF, GG]])
df = pd.DataFrame(cf, columns=['Failure', 'Normal'], index=['Failure', 'Normal'])
plot_confusion_matrix(df)

TP = FF
FN = FG
FP = GF
TN = GG

precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)

print("precision:", precision)
print("recall:", recall)
print("f1 score:", F1)
