from __future__ import print_function
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow logging
from config.arg_parser import parameter_parser
import tensorflow as tf
from tensorflow.keras.layers import Normalization, Concatenate, Reshape, Flatten, Dropout, Dense, Input, Layer
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from config.attention.Custom_Attention import CustomAttention
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np



# Set print options to display the entire array
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

args = parameter_parser()

class WIDENNET_Attention:
    def __init__(self, data, args):
        # self.name = args.mv
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr

        self.vectors = np.stack(data.iloc[:, 0].values)
        self.labels = data.iloc[:, 1].values

        positive_idxs = np.where(self.labels == 1)[0]
        negative_idxs = np.where(self.labels == 0)[0]

        idxs = np.concatenate([positive_idxs, negative_idxs])

        x_train, x_test, y_train, y_test = train_test_split(self.vectors[idxs], self.labels[idxs],
                                                            test_size=0.2, stratify=self.labels[idxs], random_state=42)

        self.x_train_wide, self.x_train_deep = x_train[:, :50], x_train[:, 50:]

        self.x_test_wide, self.x_test_deep = x_test[:, :50], x_test[:, 50:]

        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)

        classes = np.array([0, 1])
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.labels)
        self.class_weight = {index: weight for index, weight in enumerate(class_weights)}

        input_deep = Input(shape=(self.x_train_deep.shape[1], self.x_train_deep.shape[2]))
        input_wide = Input(shape=(self.x_train_wide.shape[1], self.x_train_wide.shape[2]))

        self.model = self.build_model(inputs=[input_wide, input_deep])

    def build_model(self, inputs):
        wide = Normalization()(inputs[0]) # normalize the wide input tensor
        deep = Normalization()(inputs[1]) # normalize the deep input tensor

        # subject the deep component through hidden layers
        hidden_layer1 = Dense(192, activation='relu')(deep) # 192 256
        hidden_layer2 = Dense(320, activation='relu')(hidden_layer1) #320 352
        # apply attention layer to deep component
        attention = CustomAttention(units=8)(hidden_layer2) # 16 48
        # merge the wide and deep components
        merged = Concatenate(axis=-1)([wide, attention])
        flattened = Flatten()(merged)
        output = Dense(2, activation='softmax')(flattened)

        model = Model(inputs=inputs, outputs=output)

        optimizer = Adam(self.lr)
        # optimizer = SGD(self.lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        self.model.fit(
            ([self.x_train_deep, self.x_train_wide]), self.y_train,
            epochs=self.epochs, class_weight=self.class_weight, verbose=1, batch_size=self.batch_size
        )
        # self.models.save_weights(self.name + "_model.pkl")

    def test(self):
        # self.models.load_weights(self.name + "_model.pkl")
        values = self.model.evaluate([self.x_test_deep, self.x_test_wide], self.y_test, batch_size=self.batch_size, verbose=0)
        print("\nAccuracy: ", values[1])
        predictions = (self.model.predict([self.x_test_deep, self.x_test_wide], batch_size=self.batch_size, verbose=0)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall: ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
