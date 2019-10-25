import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd


# KSOM complete code

# # INPUT ###############
# Dataset :
# topology of map : Rectangular grid or hexagonal grid
# size of map : Size of neuron map

# # OUPUT ###############
# Weights (codebook) of each output nodes


class Ksom(object):
    """Kohonen self organizing map implementation

    Attributes :
        topo : Topology of map
        size : Size of neuron maps
    """

    def __init__(self, size, topo="rect"):
        self.size = size
        self.topo = topo
        self._initialize()

    def _initialize(self):
        if self.topo == "rect":
            pass
        elif self.topo == "hex":
            pass
        else:
            pass

    def train(self, X, Y, clust, epochs, theta=2, learn_rate=0.7):
        """Train the model giving the input data

        Attributes :
            X : input dataset for training
        """
        # TODO - check for input shape
        # TODO - CHeck for basic training steps

        # TODO - the code is only for 2d dataset, change it for general
        # num_vector is number of input vectors
        num_vectors = X.shape[0]
        # num_features is number of input features of each vectors
        num_features = X.shape[1]
        self._train(num_vectors, num_features, X, Y, clust, epochs, theta, learn_rate)

    def _train(self, num_vectors, num_features, X, Y, clust, epochs, theta, learn_rate):

        # 1 Initialization of weights
        neuron_weights = np.random.rand(self.size, self.size, num_features)
        neuron_winner = np.zeros((self.size, self.size), dtype='int')
        # plt.imshow(neuron_weights, interpolation='none')
        # plt.show()
        for epoch in range(epochs):
            print("trainng ..... ", epoch)
            n = 0
            while (n < (3 * num_vectors)):
                # 2 randomly select one input vector
                curr_vector_num = random.randint(0, num_vectors - 1)
                # curr_vector_num = n
                c_iteration = epoch
                t_iteration = epochs
                map_radius = float(10)

                # 3 Find best weight vector
                min_dist = self._euclidean_dist(X[curr_vector_num],
                                                neuron_weights[0][0], num_features)
                best_neuron = [0, 0]
                for i in range(self.size):
                    for j in range(self.size):
                        dist = self._euclidean_dist(X[curr_vector_num],
                                                    neuron_weights[i][j], num_features)
                        if dist < min_dist:
                            min_dist = dist
                            best_neuron = [i, j]

                neuron_winner[best_neuron[0]][best_neuron[1]] = neuron_winner[best_neuron[0]][best_neuron[1]] + 1
                # 4 finding neighborhood neurons
                _lambda = t_iteration / map_radius
                theta_t = theta * (math.exp(-(c_iteration / _lambda)))

                neighborhood_neurons = self._neighborhood_neurons(theta_t, best_neuron)

                # 5 updating weights of neighborhood neurons
                learn_rate_t = learn_rate * (math.exp(-c_iteration / _lambda))
                for e, cn_neurons in enumerate(neighborhood_neurons):
                    omega_t = math.exp(-((e) ** 2) / (2 * theta_t ** 2))
                    for neuron in cn_neurons:
                        i = neuron[0]
                        j = neuron[1]
                        for l in range(num_features):
                            neuron_weights[i][j][l] = neuron_weights[i][j][l] + \
                                                      (X[curr_vector_num][l] - neuron_weights[i][j][l]) * \
                                                      (learn_rate_t) * omega_t

                n += 1
                # print neuron_weights[best_neuron[0]][best_neuron[1]][0] - previous_neuron_w[best_neuron[0]][best_neuron[1]][0]

        print(neuron_winner)

        # clustering for given number of classes
        neuron_class = np.zeros((self.size, self.size), dtype='int')
        neuron_clust = [[0, 0, 0] for a in range(clust)]

        for i in range(self.size):
            for j in range(self.size):
                done_for_1neuron = False
                for clst in range(clust):
                    if done_for_1neuron == False:
                        if neuron_winner[i][j] > neuron_clust[clst][0]:
                            u = clust - 1
                            while (u > clst):
                                neuron_clust[u][0] = neuron_clust[u - 1][0]
                                neuron_clust[u][1] = neuron_clust[u - 1][1]
                                neuron_clust[u][2] = neuron_clust[u - 1][2]

                                u -= 1

                            neuron_clust[clst][0] = neuron_winner[i][j]
                            neuron_clust[clst][1] = i
                            neuron_clust[clst][2] = j
                            done_for_1neuron = True

        print(neuron_clust)

        # clustering...
        for i in range(self.size):
            for j in range(self.size):
                best_class = 0
                min_edist = self._euclidean_dist(neuron_weights[i][j],
                                                 neuron_weights[neuron_clust[0][1]][neuron_clust[0][2]],
                                                 num_features)
                for clst in range(1, clust):
                    temp_dist = self._euclidean_dist(neuron_weights[i][j],
                                                     neuron_weights[neuron_clust[clst][1]][neuron_clust[clst][2]],
                                                     num_features)
                    if temp_dist < min_edist:
                        best_class = clst
                        min_edist = temp_dist
                neuron_class[i][j] = best_class

        print(neuron_class)

        # plt.imshow(neuron_weights, interpolation='none')
        # plt.show()

        cY = np.zeros((num_vectors), dtype='int')
        for vec in range(num_vectors):
            min_dist = self._euclidean_dist(X[curr_vector_num],
                                            neuron_weights[0][0], num_features)
            best_neuron = [0, 0]
            for i in range(self.size):
                for j in range(self.size):
                    dist = self._euclidean_dist(X[curr_vector_num],
                                                neuron_weights[i][j], num_features)
                    if dist < min_dist:
                        min_dist = dist
                        best_neuron = [i, j]
            cY[vec] = neuron_class[best_neuron[0]][best_neuron[1]]
        print(cY, Y)

        # self._check_accuracy(cY, Y, clust)

    def _euclidean_dist(self, curr_vector, weight_vector, num_features):
        total = float(0)

        for i in range(num_features):
            diff = curr_vector[i] - weight_vector[i]
            total = total + diff ** 2

        return math.sqrt(total)

    def _neighborhood_neurons(self, theta_t, best_neuron):
        # TODO - improve algorithm
        curr_radios = math.floor(theta_t)
        t = 0
        i = best_neuron[0]
        j = best_neuron[1]
        n_neuron = [[[i, j]]]
        while (t < curr_radios):
            cn_neuron = []
            i1 = i - (t + 1)
            i2 = i + (t + 1)
            j1 = j - (t + 1)
            j2 = j + (t + 1)
            for q in range(2 * (t + 1) + 1):
                i1 = i1 + q
                if self._check_neuron_position(i1, j1):
                    cn_neuron.append([i1, j1])
            i1 = i - (t + 1)
            for q in range(1, 2 * (t + 1) + 1):
                j1 = j1 + q
                if self._check_neuron_position(i1, j1):
                    cn_neuron.append([i1, j1])
            j1 = j - (t + 1)
            for q in range(2 * (t + 1)):
                i2 = i2 - q
                if self._check_neuron_position(i2, j2):
                    cn_neuron.append([i2, j2])
            i2 = i + (t + 1)
            for q in range(1, 2 * (t + 1)):
                j2 = j2 - q
                if self._check_neuron_position(i2, j2):
                    cn_neuron.append([i2, j2])

            t += 1
            n_neuron.append(cn_neuron)
        return n_neuron

    def _check_neuron_position(self, i, j):
        if i >= 0 and i < self.size:
            if j >= 0 and j < self.size:
                return True
        return False

    def _check_accuracy_basic(self, cY, Y):
        correct = 0
        total = 0
        for e, clas in enumerate(Y):
            total += 1
            if clas == cY[e]:
                correct += 1
        return float(correct) / total

    def _check_accuracy(self, cY, Y, clust):
        false = 0
        total = len(Y)

        acc_score = [[0 for a in range(clust)] for a in range(clust)]
        for e, y in enumerate(cY):
            acc_score[y][Y[e] - 1] += 1

        for acc in acc_score:
            temp_sum = 0
            for a in acc:
                temp_sum += a
            false += (temp_sum - max(acc))

        print(false, float(false) / total)
        print(acc_score)


if __name__ == "__main__":
    df = pd.read_csv('winequality-white.csv', sep=';')
    x = np.array(df.drop(['quality'], axis=1), dtype='float32')
    y = np.array(df['quality'], dtype='int32')
    # x = np.random.rand(100, 3)
    # plt.imshow(x, interpolation='none')
    # plt.show()
    testksom = Ksom(20)
    testksom.train(x, y, clust=9, epochs=5)

    while (1):
        pass