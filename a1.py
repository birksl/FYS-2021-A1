import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SF_CSV = "data/SpotifyFeatures.csv"


def prepare_data(plot: bool):
    # Read data
    df = pd.read_csv(SF_CSV)

    # Drop columns we aren't interested in
    df = df[["genre", "liveness", "loudness"]]

    # Filter for only Pop and Classical songs
    df = df[(df.genre == "Pop") | (df.genre == "Classical")]
    # Add label column with Pop = 1 and Classical = 0
    df["label"] = (df.genre == "Pop") * 1

    # Drop genre column
    df = df[["label", "liveness", "loudness"]]
    df = df.sort_values(by=["label"])

    # Convert to matrix and split into two by label
    dataset = df.to_numpy()
    classics = len(df[df.label == 0])
    pops = len(df[df.label == 1])

    # Split each class matrix into two with a ration of 80/20
    # Concatenate the 80 parts back together and the 20s back together
    # We now have a training and test dataset with equal proportions of both classes
    training = np.concatenate(
        (
            dataset[0 : int(classics * 0.8)],
            dataset[classics : classics + int(pops * 0.8)],
        )
    )
    test = np.concatenate(
        (
            dataset[0 : int(classics * 0.2)],
            dataset[classics : classics + int(pops * 0.2)],
        )
    )
    # We shuffle the training set
    np.random.shuffle(training)
    # Extract labels as it's own vector
    training_labels = training.transpose()[0]
    training = training.transpose()[1:].transpose()

    test_labels = test.transpose()[0]
    test = test.transpose()[1:].transpose()

    if plot:
        # We scatterplot loudness vs liveness and label them with the classification
        plt.scatter(
            training[training_labels == 0].transpose()[0],
            training[training_labels == 0].transpose()[1],
            label="Classical",
        )
        plt.scatter(
            training[training_labels == 1].transpose()[0],
            training[training_labels == 1].transpose()[1],
            label="Pop",
            alpha=0.5,
        )
        plt.legend()
        plt.xlabel("liveness")
        plt.ylabel("loudness")
        plt.show()
    return training, training_labels, test, test_labels


# Sigmoid function to convert linear output to probabilities
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Function to compute the loss (Binary Cross Entropy)
def compute_loss(labels, prediction):
    # Clip prediction to avoid log(0) which results in divide by zero error
    prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
    return -np.mean(labels * np.log(prediction) + (1 - labels) * np.log(1 - prediction))


class LogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = []

    def train(self, samples, labels):
        self.weights = np.zeros(samples.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range(len(samples)):
                z = np.dot(samples[i], self.weights) + self.bias

                # Predict value
                prediction = sigmoid(z)

                # Compute the gradient for weights and bias
                error = prediction - labels[i]
                dw = error * samples[i]  # Derivative with respect to weights
                db = error  # Derivative with respect to bias

                # Update weights and bias using the gradient and learning rate
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Compute the loss for the entire dataset at the end of each epoch
            complete_prediction = sigmoid(np.dot(samples, self.weights) + self.bias)
            loss = compute_loss(labels, complete_prediction)
            self.loss.append(loss)  # Store the loss for plotting

    def plot_loss(self, name):
        plt.figure()
        plt.plot(self.loss)
        plt.title("Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.savefig(name)

    def predictions(self, prediction_samples):
        """Use the trained weights to predict a new set of samples"""
        z = np.dot(prediction_samples, self.weights) + self.bias
        y_pred = sigmoid(z)
        return np.where(y_pred >= 0.5, 1, 0)

    def accuracy(self, prediction_samples, labels):
        predictions = self.predictions(prediction_samples)
        success = np.sum(labels == predictions)
        return success / len(predictions)

    def plot_decision_boundary(self, samples, labels, save_name: str):
        plt.figure()

        plt.scatter(
            samples[labels == 0].transpose()[0],
            samples[labels == 0].transpose()[1],
            label="Classical",
        )
        plt.scatter(
            samples[labels == 1].transpose()[0],
            samples[labels == 1].transpose()[1],
            label="Pop",
            alpha=0.5,
        )

        # Extract learned weights and bias
        w0, w1 = self.weights
        b = self.bias

        # Define x values (liveness)
        x_values = np.linspace(0, 1, 100)

        # Compute the corresponding y values (loudness) using the equation of the decision boundary
        y_values = -(w0 / w1) * x_values - b / w1

        # Plot the decision boundary
        plt.plot(x_values, y_values, color="black", label="Decision Boundary")
        plt.xlim(0, 1)
        plt.title("Dataset with Decision Boundary")
        plt.xlabel("Liveness")
        plt.ylabel("Loudness")
        plt.legend()
        if not save_name:
            plt.show()
        else:
            plt.savefig(save_name)


def confusion_matrix(labels, predictions):
    # Initialize the confusion matrix
    matrix = np.zeros((2, 2), dtype=int)

    # Calculate True Negatives, False Positives, False Negatives, and True Positives
    matrix[0, 0] = np.sum((labels == 0) & (predictions == 0))  # True Negative (TN)
    matrix[0, 1] = np.sum((labels == 0) & (predictions == 1))  # False Positive (FP)
    matrix[1, 0] = np.sum((labels == 1) & (predictions == 0))  # False Negative (FN)
    matrix[1, 1] = np.sum((labels == 1) & (predictions == 1))  # True Positive (TP)

    return matrix


def plot_confusion_matrix(conf_matrix, name):
    plt.figure()
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Classical (0)", "Pop (1)"],
        yticklabels=["Classical (0)", "Pop (1)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(name)


def run():
    training, labels, test, test_labels = prepare_data(False)
    model = LogisticRegression(0.0005, 100)
    model.train(training, labels)
    print(model.weights, model.bias)
    accuracy = model.accuracy(training, labels)
    print(accuracy)
    model.plot_loss(f"loss_{model.learning_rate}_{model.epochs}.jpeg")
    model.plot_decision_boundary(
        training,
        labels,
        f"boundary_{model.learning_rate}_{model.epochs}.jpeg",
    )
    accuracy = model.accuracy(test, test_labels)
    print(accuracy)
    plot_confusion_matrix(
        confusion_matrix(test_labels, model.predictions(test)),
        f"conf_{model.learning_rate}_{model.epochs}.jpeg",
    )


if __name__ == "__main__":
    run()
