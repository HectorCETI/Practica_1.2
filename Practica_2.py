import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, KFold


class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, max_epochs=1000):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # dot product + bias
        return 1 if summation > 0 else -1  # Assuming binary classification

    def train(self, training_inputs, labels):
        for _ in range(self.max_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


def generate_modified_data(data):
    modified_data = data.copy()
    modified_data.iloc[:, 3][modified_data.iloc[:, 0] == -1] = 1
    return modified_data


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.iloc[:, :-1].values, data.iloc[:, -1].values


def evaluate_perceptron(train_data, train_labels, test_data, test_labels):
    perceptron = Perceptron(input_size=train_data.shape[1])
    perceptron.train(train_data, train_labels)
    return perceptron


def main():
    file_paths = ["spheres2d10.csv", "spheres2d50.csv", "spheres2d70.csv"]
    perturbation_levels = [10, 50, 70]

    for file_path, perturbation_level in zip(file_paths, perturbation_levels):
        print(f"Processing file: {file_path}")

        # Cargar datos
        data = pd.read_csv(file_path, header=None)

        # Generar conjunto de datos modificado
        modified_data = generate_modified_data(data)

        # Dividir conjunto de datos en entrenamiento y prueba
        train_data, test_data, train_labels, test_labels = train_test_split(
            modified_data.iloc[:, :-1], modified_data.iloc[:, -1], test_size=0.2, random_state=42
        )

        # Entrenar el perceptrón
        perceptron = evaluate_perceptron(train_data.values, train_labels.values, test_data.values, test_labels.values)

        # Visualizar los datos en 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_data.iloc[:, 0], train_data.iloc[:, 1], train_data.iloc[:, 2], c=train_labels)

        # Calcular plano separador
        if perceptron.weights[1] != 0:
            xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
            z = (-perceptron.weights[0] - perceptron.weights[1] * xx - perceptron.weights[2] * yy) / perceptron.weights[
                3]
            ax.plot_surface(xx, yy, z, alpha=0.5, rstride=100, cstride=100)

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.set_title(f"Perceptron Separating Plane for {perturbation_level}% perturbation")

        plt.show()


if __name__ == "__main__":
    main()
