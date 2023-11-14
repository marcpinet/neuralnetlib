from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from neuralnetlib.model import Model
from neuralnetlib.layers import Dense, Activation
from neuralnetlib.activations import Linear, LeakyReLU
from neuralnetlib.losses import MeanSquaredError, MeanAbsoluteError
from neuralnetlib.optimizers import Adam

def main():
    # 1. Loading a dataset (in this case, the diabetes dataset)
    x, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 2. Preprocessing the dataset
    scaler_x = MinMaxScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # 3. Model definition
    input_neurons = x_train.shape[1:][0]
    num_hidden_layers = 2
    hidden_neurons = 2
    output_neurons = 1

    model = Model()
    model.add(Dense(input_neurons, hidden_neurons, weights_init='he', random_state=42))
    model.add(Activation(LeakyReLU()))

    for _ in range(num_hidden_layers - 1):
        model.add(Dense(hidden_neurons, hidden_neurons, weights_init='he', random_state=42))
        model.add(Activation(LeakyReLU()))

    model.add(Dense(hidden_neurons, output_neurons, random_state=42))
    model.add(Activation(Linear()))

    # 4. Model compilation
    model.compile(loss_function=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))

    # 5. Model training
    model.train(x_train, y_train, epochs=10, batch_size=32, random_state=42)

    # 6. Model evaluation
    loss = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss}', "function=" + model.loss_function.__class__.__name__)

    # 7. Model prediction and a loss metric (specific to regression)
    y_pred = model.predict(x_test)
    print("MAE: ", MeanAbsoluteError()(y_test, y_pred))

    # 8. We won't print metrics such as accuracy or f1-score because this is a regression problem
    # not a classification-regression one.

if __name__ == '__main__':
    main()