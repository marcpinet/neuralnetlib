import numpy as np
import pandas as pd
import requests
from scipy.io import arff
from sklearn.model_selection import train_test_split

from neuralnetlib.activations import ReLU, Sigmoid
from neuralnetlib.callbacks import EarlyStopping
from neuralnetlib.layers import Input, Dense, Activation
from neuralnetlib.losses import BinaryCrossentropy
from neuralnetlib.metrics import accuracy_score
from neuralnetlib.models import Sequential
from neuralnetlib.optimizers import Adam


def main():
    # 1. Get the Tic Tac Toe dataset
    link = "https://www.openml.org/data/download/50/dataset_50_tic-tac-toe.arff"
    file_path = "tic-tac-toe.arff"
    r = requests.get(link, allow_redirects=True)
    with open(file_path, 'wb') as f:
        f.write(r.content)

    # 2. Load the dataset
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # 3. Data preprocessing
    char_map = {'b\'x\'': 1, 'b\'o\'': -1,
                'b\'b\'': 0}  # We replace the 'x', 'o', 'b' values with 1, -1, 0 from the arff file
    for col in df.columns[:-1]:
        df[col] = df[col].map(lambda x: char_map.get(str(x), x))

    class_map = {'b\'positive\'': 1,
                 'b\'negative\'': 0}  # We replace the 'positive', 'negative' values with 1, 0 from the arff file
    df['Class'] = df['Class'].map(lambda x: class_map.get(str(x), x))

    # 4. Create the dataset and split it into train and test sets
    X = df.drop(columns=['Class'])
    y = df['Class']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # 5. Create the model
    input_neurones = x_train.shape[1:][0]
    hidden_neurones = 64
    hidden_layers = 2
    output_neurones = 1

    # Print the shapes of the train and test sets
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    model = Sequential()
    model.add(Input(input_neurones))
    model.add(Dense(hidden_neurones, weights_init='he', random_state=42))
    model.add(Activation(ReLU()))
    for _ in range(hidden_layers):
        model.add(Dense(hidden_neurones, weights_init='he', random_state=42))
        model.add(Activation(ReLU()))

    model.add(Dense(output_neurones, random_state=42))
    model.add(Activation(Sigmoid()))

    # 6. Compile the model
    model.compile(loss_function=BinaryCrossentropy(), optimizer=Adam(learning_rate=0.001))

    # 7. Train the model
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=500, batch_size=32, metrics=[accuracy_score], random_state=42,
              callbacks=[early_stopping])

    # 8. Evaluate the model
    loss, preds = model.evaluate(x_test, y_test)
    print(f"Loss: {loss}")

    # 9. Make predictions on test set and print the accuracy
    y_pred = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_pred, y_test)}")

    # 10. (Optionally) Delete the file
    import os
    os.remove(file_path)

    # 11. Save the model
    model.save("tic_tac_toe_model.npz")


if __name__ == "__main__":
    main()
