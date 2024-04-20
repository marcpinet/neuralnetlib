import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

from neuralnetlib.activations import ReLU, Softmax
from neuralnetlib.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation
from neuralnetlib.losses import CategoricalCrossentropy
from neuralnetlib.metrics import accuracy_score, f1_score, recall_score
from neuralnetlib.model import Model
from neuralnetlib.optimizers import Adam
from neuralnetlib.utils import one_hot_encode


def main():
    # 1. Loading a dataset (in this case, MNIST)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Preprocessing
    x_train = x_train.reshape(-1, 1, 28, 28) / 255.0  # Normalization and reshaping of the images for CNN
    x_test = x_test.reshape(-1, 1, 28, 28) / 255.0  # Normalization and reshaping of the images for CNN
    y_train = one_hot_encode(y_train, num_classes=10)  # One-hot encoding of the labels
    y_test = one_hot_encode(y_test, num_classes=10)  # One-hot encoding of the labels

    # 3. Model definition
    model = Model()
    model.add(Input(input_shape=(1, 28, 28)))
    model.add(Conv2D(filters=4, kernel_size=2, random_state=42))
    model.add(Activation(ReLU()))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=8, kernel_size=2, random_state=42))
    model.add(Activation(ReLU()))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, random_state=42))
    model.add(Activation(ReLU()))
    model.add(Dense(10, random_state=42))
    model.add(Activation(Softmax()))
    """
        Side note: if you set the following:
        
          - filters to 8 and 16 (in this order)
          - padding of the Conv2D layers to 'same'
          - weights initialization to 'he'
        
        you'll get an accuracy of ~0.9975 which is actually pretty cool
    """
    # 4. Model compilation
    model.compile(loss_function=CategoricalCrossentropy(), optimizer=Adam())

    # 5. Model training
    model.train(x_train, y_train, epochs=10, batch_size=128, metrics=[accuracy_score], random_state=42,
                validation_data=(x_test, y_test))

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
    model.save("my_mnist_cnn_model.npz")


if __name__ == '__main__':
    main()
