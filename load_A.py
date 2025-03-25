import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os


def load_dict_A(folder="./"):
    """
    :param folder: The path to the folder containing the file spectra_USGS_ices_v2.mat. It is "./" (i.e. current working directory) by default.
    :return: A the spectra dictionary, N the number of points for a spectrum, P the number of spectra, and the wavelengths corresponding to the data.
    """
    Dic = loadmat(os.path.join(folder,"spectra_USGS_ices_v2.mat"))
    A = Dic['speclib'] # model dictionary, it contains, column-wise, the endmembers
    N, P =  A.shape
    wavelengths = Dic['wavelength'][:,0] # x-axis for the spectra (y-axis is A[:,i]: it yields the reflectance for spectrum i)
    return A, N, P, wavelengths


def plot_spectra(A, L_indices, wavelengths):
    """
    :param A: The dictionary.
    :param L_indices: A list or array of indices for endmember selection in A.
    :param wavelengths: x-axis values.
    """
    colors = list(mpl.colors.TABLEAU_COLORS) # list of colors used so that a given index (endmember) will always be the same color
    plt.figure(figsize = (10,6))
    for p in L_indices:
        plt.plot(wavelengths, A[:, p], colors[p % len(colors)], label="Spectrum %d" % p)
    plt.title("Some spectra (indices: %s)"%L_indices)
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Amplitude")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    pass


def run_exemple1():
    """
    Load the dictionary and plot 5 endmembers randomly.
    """
    A, N, P, wavelengths = load_dict_A()
    L_indices_random = np.random.randint(0, P, size=5)
    plot_spectra(A, L_indices_random, wavelengths)
    pass


run_exemple1()