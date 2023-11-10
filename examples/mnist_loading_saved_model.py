from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

from neuralnetlib.model import Model
from neuralnetlib.utils import one_hot_encode
from neuralnetlib.metrics import accuracy_score


def main():
    # 1. Loading the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Preprocessing
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    y_train = one_hot_encode(y_train, num_classes=10)
    y_test = one_hot_encode(y_test, num_classes=10)

    # 3. Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # 4. Load the model
    model: Model = Model.load('my_mnist_model.npz')

    # 5. Predict and evaluate on the validation set
    y_pred_val = model.predict(x_val)
    accuracy_val = accuracy_score(y_pred_val, y_val)
    print(f'Validation Accuracy: {accuracy_val}')

    # 6. Optionally, you can still evaluate on the test set
    y_pred_test = model.predict(x_test)
    accuracy_test = accuracy_score(y_pred_test, y_test)
    print(f'Test Accuracy: {accuracy_test}')


if __name__ == '__main__':
    main()