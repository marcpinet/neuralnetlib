import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

from neuralnetlib.activations import Sigmoid, Softmax
from neuralnetlib.layers import Dense, Activation
from neuralnetlib.losses import CategoricalCrossentropy
from neuralnetlib.metrics import accuracy_score, f1_score, recall_score
from neuralnetlib.model import Model
from neuralnetlib.optimizers import SGD
from neuralnetlib.utils import one_hot_encode


def main():
    # 1. Loading a dataset (in this case, MNIST)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Preprocessing
    x_train = x_train.reshape(-1, 28 * 28) / 255.0  # Normalization and flattening of the images
    x_test = x_test.reshape(-1, 28 * 28) / 255.0  # Normalization and flattening of the images
    y_train = one_hot_encode(y_train, num_classes=10)  # One-hot encoding of the labels
    y_test = one_hot_encode(y_test, num_classes=10)  # One-hot encoding of the labels

    # 3. Model definition
    input_neurons = x_train.shape[1:][0]  # MNIST images are 28x28
    num_hidden_layers = 2  # Number of hidden layers
    hidden_neurons = 30  # Number of neurons in each hidden layer
    output_neurons = 10  # Assuming 10 classes for MNIST

    model = Model()
    model.add(Dense(input_neurons, hidden_neurons, weights_init='lecun', random_state=42))  # First hidden layer
    model.add(Activation(Sigmoid()))  # ...and its function activation

    for _ in range(num_hidden_layers - 1):  # Add the rest of the hidden layers
        model.add(Dense(hidden_neurons, hidden_neurons, weights_init='lecun',
                        random_state=42))  # Hidden layer must have the same number of neurons as the previous one
        model.add(Activation(Sigmoid()))  # ...and its function activation

    model.add(Dense(hidden_neurons, output_neurons, random_state=42))  # Output layer
    model.add(Activation(Softmax()))  # ...and its function activation

    # 4. Model compilation
    model.compile(loss_function=CategoricalCrossentropy(), optimizer=SGD(learning_rate=0.1))

    # 5. Model training
    model.train(x_train, y_train, epochs=20, batch_size=48, metrics=[accuracy_score], random_state=42)

    # 6. Model evaluation
    loss = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss}')

    # 7. Model prediction
    y_pred = model.predict(x_test)

    # 8. Print some metrics
    print("accuracy:", accuracy_score(y_pred, y_test))
    print("f1_score:", f1_score(y_pred, y_test))
    print("recall_score", recall_score(y_pred, y_test))

    # 8.5.  Plot the first 10 test images, their predicted labels, and the true labels.
    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        ax = fig.add_subplot(5, 2, i + 1, xticks=[], yticks=[])
        ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"Predicted: {np.argmax(y_pred[i])}, Actual: {np.argmax(y_test[i])}")
    plt.show()

    # 9. Save the model
    model.save("my_mnist_model.npz")


if __name__ == '__main__':
    main()
