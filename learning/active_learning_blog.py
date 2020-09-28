# copied from https://damienlancry.github.io/dbal-mnist/
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_keras_model():
    model = Sequential()
    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


classifier = KerasClassifier(create_keras_model)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32') / 255.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# initial labelled data
initial_idx = np.array([], dtype=np.int)
for i in range(10):
    idx = np.random.choice(np.where(y_train[:, i] == 1)[0], size=2, replace=False)
    initial_idx = np.concatenate((initial_idx, idx))

X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]

# initial unlabelled data
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)


#Query
def uniform(learner, X, n_instances=1):
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]


def max_entropy(learner, X, n_instances=1, T=100):
    random_subset = np.random.choice(X.shape[0], 2000, replace=False)
    MC_output = K.function([learner.estimator.model.layers[0].input, K.learning_phase()],
                           [learner.estimator.model.layers[-1].output])
    learning_phase = True
    MC_samples = [MC_output([X[random_subset], learning_phase])[0] for _ in range(T)]
    MC_samples = np.array(MC_samples)
    expected_p = np.mean(MC_samples, axis=0)
    acquisition = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=1)
    idx = (- acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]


# active learning procedure
def active_learning_procedure(query_strategy, test_X, test_y, pool_X, pool_y, initial_X, initial_y, estimator,
                              epochs=50, batch_size=128, n_queries=100, n_instances=10, verbose=0):
    learner = ActiveLearner(estimator=estimator, X_training=initial_X, y_training=initial_y,
                            query_strategy=query_strategy, verbose=verbose)
    perf_hist = [learner.score(test_X, test_y, verbose=verbose)]
    for index in range(n_queries):
        query_idx, query_instance = learner.query(pool_X, n_instances)
        learner.teach(pool_X[query_idx], pool_y[query_idx], epochs=epochs, batch_size=batch_size, verbose=verbose)
        pool_X = np.delete(pool_X, query_idx, axis=0)
        pool_y = np.delete(pool_y, query_idx, axis=0)
        model_accuracy = learner.score(test_X, test_y, verbose=0)
        print("accuracy after query {n}: {acc:0.4f".format(n=index + 1, acc=model_accuracy))
        perf_hist.append(model_accuracy)
    return perf_hist


estimator = KerasClassifier(create_keras_model)
entropy_perf_hist = active_learning_procedure(max_entropy, X_test, y_test, X_pool, y_pool, X_initial, y_initial, estimator,)

estimator = KerasClassifier(create_keras_model)
uniform_perf_hist = active_learning_procedure(uniform, X_test, y_test, X_pool, y_pool, X_initial, y_initial, estimator, )





