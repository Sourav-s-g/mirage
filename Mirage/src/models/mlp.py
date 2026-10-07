"""MLP baseline on hand-made image features (not on pixels batches, so it is not part of get_model)."""
from functools import lru_cache

import numpy as np
from torch import nn

N_HIST_BINS = 32
N_RADIAL_BINS = 64
THUMB_SIZE = 32


def build(in_dim, hidden=(256, 128), dropout=0.3):
    """Two hidden layers with dropout, one logit out of shape (B, 1)."""
    layers = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def color_histogram(img):
    """Per-channel histogram with 32 bins, as fractions of the pixel count."""
    arr = np.asarray(img)
    bin_width = 256 // N_HIST_BINS
    hists = [np.bincount((arr[:, :, c] // bin_width).ravel(), minlength=N_HIST_BINS)
             for c in range(3)]
    return np.concatenate(hists) / (arr.shape[0] * arr.shape[1])


@lru_cache(maxsize=None)
def radial_bins(h, w):
    """Radial bin index of every pixel of a centred spectrum, and the pixel count per bin."""
    y, x = np.indices((h, w))
    r = np.hypot(y - h // 2, x - w // 2)
    r_max = min(h, w) / 2
    bins = (r / r_max * N_RADIAL_BINS).astype(int)
    # corners lie outside the largest circle, they go into an extra bin that is dropped
    bins = np.minimum(bins, N_RADIAL_BINS)
    counts = np.bincount(bins.ravel(), minlength=N_RADIAL_BINS + 1)[:N_RADIAL_BINS]
    return bins, np.maximum(counts, 1)


def radial_log_spectrum(img):
    """Log power spectrum of the grayscale image, averaged over rings of equal frequency."""
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    power = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    log_power = np.log(power + 1e-8)
    bins, counts = radial_bins(*gray.shape)
    sums = np.bincount(bins.ravel(), weights=log_power.ravel(),
                       minlength=N_RADIAL_BINS + 1)[:N_RADIAL_BINS]
    return sums / counts


def spectrum_hist_features(img):
    return np.concatenate([radial_log_spectrum(img), color_histogram(img)])


def pixel_features(img):
    thumb = img.convert("L").resize((THUMB_SIZE, THUMB_SIZE))
    return np.asarray(thumb, dtype=np.float64).ravel() / 255


FEATURES = {"spectrum_hist": spectrum_hist_features, "pixels": pixel_features}
