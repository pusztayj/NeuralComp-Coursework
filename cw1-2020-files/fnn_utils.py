
# Plotting and data reading functions for Feedforward Neural Networks.

import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

fig, axarr = plt.subplots(3, figsize=(6, 8))
plt.subplots_adjust(wspace=1, hspace=1)
fig.show()

nx, ny = 5, 5
fig_p, axarr_p = plt.subplots(ny, nx, figsize=(7.5, 7.5))
plt.subplots_adjust(wspace=1, hspace=1)
fig_p.show()


def display_predictions(network, show_pct=False):
    fashion = ["T-shirt", "Trouser", "Pullover", "Dress",
               "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
    permutation = np.random.permutation(len(network.testX))
    for y in range(ny):
        for x in range(nx):
            i = permutation[x*ny+y]
            predicted = network.predict(network.testX[i])
            correct = np.argmax(network.testY[i])
            if predicted == correct:
                color = "Greys"
            else:
                color = "Reds"
            label = fashion[predicted]
            if show_pct:
                label += " (" + \
                    '{0:.2f}'.format(network.predict_pct(predicted)) + ")"
            axarr_p[y, x].set_xlabel(label, fontsize=10)
            axarr_p[y, x].imshow(network.testX[i].reshape(28, 28), cmap=color)
            axarr_p[y, x].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')  # labels along the bottom edge are off
            axarr_p[y, x].axes.get_yaxis().set_visible(False)
    fig_p.canvas.draw()
    plt.pause(0.01)


def plot_stats(a, loss, test_acc_log, train_acc_log):

    L = len(a)

    # Plot accuracies
    axarr[0].clear()
    axarr[0].set_title("Classification accuracy", fontsize=18)
    axarr[0].plot(test_acc_log)
    axarr[0].plot(train_acc_log)
    axarr[0].legend(["Test", "Training"])

    # Plot activations
    axarr[1].clear()
    axarr[1].set_title("Activations", fontsize=18)
    axarr[1].violinplot(a)

    # Plot logs
    axarr[2].clear()
    axarr[2].set_title("Loss", fontsize=18)
    axarr[2].plot(loss)

    fig.canvas.draw()
    plt.pause(0.01)


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, np.eye(10)[labels]


def read_data():
    mnisttrainimages, mnisttrainlabels = load_mnist('fashion', kind='train')
    mnisttestimages, mnisttestlabels = load_mnist('fashion', kind='t10k')
    return mnisttrainimages, mnisttrainlabels, mnisttestimages, mnisttestlabels
