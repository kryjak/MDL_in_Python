import numpy as np
import matplotlib.pyplot as plt

def create_spiral(n_arms=6, n_points_per_arm=100):
    if n_arms % 2 != 0:
        raise ValueError('Number of arms should be even (this is so that we have classes of equal size).')
    radii = np.linspace(1, 10, 100)

    angle_init = np.pi/2
    angle_final = angle_init - np.pi/2

    angles = np.linspace(angle_init, angle_final, n_points_per_arm)

    X1 = radii*np.cos(angles)
    X2 = radii*np.sin(angles)

    for ii in range(1, n_arms):
        X1 = np.concatenate((X1, radii*np.cos(angles + ii*2*np.pi/n_arms)))
        X2 = np.concatenate((X2, radii*np.sin(angles + ii*2*np.pi/n_arms)))

    noise_X1 = np.random.normal(loc=0, scale=0.1, size=len(X1))
    noise_X2 = np.random.normal(loc=0, scale=0.1, size=len(X2))
    X1 = X1 + noise_X1
    X2 = X2 + noise_X2

    X = np.vstack((X1, X2)).T

    Y = np.array([0]*n_points_per_arm + [1]*n_points_per_arm)
    Y = np.tile(Y, int(n_arms/2))

    plt.scatter(X[:,0], X[:,1], marker='.', c=Y)
    plt.axis('scaled')
    plt.show()

    return X, Y

X, Y = create_spiral(6)