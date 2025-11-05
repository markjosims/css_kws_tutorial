from typing import *
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import Normal

def mfcc_w_deltas(audio: Union[str, np.ndarray], samplerate: Optional[int]=None) -> np.ndarray:
    if type(audio) is str:
        audio, samplerate = librosa.load(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=12)
    energy = librosa.feature.rms(y=audio)
    mfcc = np.concat([mfcc, energy], axis=0)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    feature = np.concat([mfcc, d1, d2], axis=0)
    return feature

def fit_gaussians(
        X: np.ndarray,
        y: np.ndarray,
        phones: Union[str, Sequence[str]]='ailn',
        n_dist: int=4,
    ) -> List[GeneralMixtureModel]:
    """
    Arguments:
        X: Data points (num_features x num_samples).
        Y: Labels for each data point in X.
        phones: List of phonemes corresponding to each state.
        n_dist: Number of Gaussian distributions to fit for each phoneme.
    Returns:
        List of Gaussian distributions fitted to the data for each phoneme.
    """
    states = []
    for phone in phones:
        phone_X = X[y==phones.index(phone)]
        if n_dist==1:
            state = Normal()
        else:
            state = GeneralMixtureModel([Normal() for _ in range(n_dist)])
        state.fit(phone_X)
        states.append(state)
    return states

def plot_gaussians(
        states: List[GeneralMixtureModel],
        X_2d: np.ndarray,
        y: np.ndarray,
        phones: Union[str, Sequence[str]]='ailn',
        colors: Sequence[str]=['purple', 'blue', 'green', 'red'],
        ax=None,
    ):
    """
    Arguments:
        states: List of Gaussian distributions corresponding to each state.
        X: Original high-dimensional data points.
        Y: Labels for each data point in X.
        X_2d: 2D representation of the data points (e.g., via PCA or t-SNE).
        phones: List of phonemes corresponding to each state.
        colors: Colors to use for each phoneme.
    Plot the 2D Gaussian distributions for each state.
    """
    ax = ax or plt.gca()
    x_min = X_2d[:,0].min()
    x_max = X_2d[:,0].max()
    xticks = np.linspace(x_min, x_max, num=100)
    y_min = X_2d[:,1].min()
    y_max = X_2d[:,1].max()
    yticks = np.linspace(y_min, y_max, num=100)

    assert len(xticks)==100
    assert len(yticks)==100

    xx, yy = np.meshgrid(xticks, yticks)
    x_ = np.array(list(zip(xx.flatten(), yy.flatten())))

    for state, phone, color in zip(states, phones, colors):
        prob = state.probability(x_).reshape(len(xticks), len(yticks))
        phone_X = X_2d[y==phones.index(phone)]

        # only show probability above 90th quantile, to minimize overlap
        quantile90 = prob.quantile(0.90)
        prob[prob<quantile90]=float('-inf')

        ax.contourf(xx, yy, prob, cmap=color.capitalize()+'s', alpha=0.5)
        ax.scatter(phone_X[:,0], phone_X[:,1], s=10, color=color, alpha=0.2, label=phone)

def plot_mfcc_2d(
        mfcc_2d: np.ndarray,
        y: np.ndarray,
        phones: Union[str, Sequence[str]]='ailn',
        colors: Sequence[str]=['purple', 'blue', 'green', 'red'],
        ax=None,
    ):
    """
    Arguments:
        mfcc_2d: 2D representation of the MFCC data points (e.g., via PCA or t-SNE).
        Y: Labels for each data point in mfcc_2d.
        phones: List of phonemes corresponding to each state.
        colors: Colors to use for each phoneme.
    Plot the 2D MFCC data points colored by phoneme.
    """
    ax = ax or plt.gca()
    for phone, color in zip(phones, colors):
        phone_X = mfcc_2d[y==phones.index(phone)]
        ax.scatter(phone_X[:,0], phone_X[:,1], s=10, color=color, alpha=0.5, label=phone)

def plot_mfcc_w_dists(
        mfcc_2d: np.ndarray,
        y: np.ndarray,
        num_dist_seq: Sequence[int] = [0, 1, 2, 3],
        phones: Union[str, Sequence[str]]='ailn',
        colors: Sequence[str]=['purple', 'blue', 'green', 'red'],
        ax=None,      
):
    ax = plt.subplots(len(num_dist_seq), figsize=(12, 6*len(num_dist_seq)))[1]
    for i, n_dist in enumerate(num_dist_seq):
        if n_dist == 0:
            plot_mfcc_2d(
                mfcc_2d,
                y,
                phones=phones,
                colors=colors,
                ax=ax[i],
            )
            ax[i].set_title(f'MFCC 2D representation')
        else:
            states = fit_gaussians(
                mfcc_2d,
                y,
                phones=phones,
                n_dist=n_dist,
            )
            plot_gaussians(
                states,
                mfcc_2d,
                y,
                phones=phones,
                colors=colors,
                ax=ax[i],
            )
            ax[i].set_title(f'MFCC 2D with {n_dist} Gaussian(s) per phoneme')
            ax[i].legend()
    plt.tight_layout()
    return ax