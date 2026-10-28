"""Data loading (numpy only): MedMNIST datasets or MNIST; falls back to sklearn digits (upscaled to 28x28)."""
import gzip
import os
import urllib.request

import numpy as np

# On macOS, Python installed from python.org may not find the root certificates, so downloads fail with
# CERTIFICATE_VERIFY_FAILED. If certifi is installed, point Python at its certificate bundle
# (certificates are still verified normally; this only tells Python where they are).
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

_MNIST_URLS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
_FILES = {"train": "train-images-idx3-ubyte.gz", "test": "t10k-images-idx3-ubyte.gz"}


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def _try_torchvision(split, root):
    from torchvision import datasets  # noqa
    ds = datasets.MNIST(root, train=(split == "train"), download=True)
    return ds.data.numpy()


def _try_download(split, root):
    os.makedirs(root, exist_ok=True)
    fn = os.path.join(root, _FILES[split])
    if not os.path.exists(fn):
        last = None
        for base in _MNIST_URLS:
            try:
                urllib.request.urlretrieve(base + _FILES[split], fn)
                break
            except Exception as e:  # noqa
                last = e
        else:
            raise last
    return _read_idx(fn)


def _sklearn_digits():
    from sklearn.datasets import load_digits
    from skimage.transform import resize
    d = load_digits().images / 16.0  # 8x8
    out = np.zeros((len(d), 28, 28), np.float32)
    for i, im in enumerate(d):
        out[i, 4:24, 4:24] = resize(im, (20, 20), order=1, anti_aliasing=False)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def load_digits_28(split="train", root="./data"):
    """Returns (N,28,28) float32 images in [0,1]."""
    for fn in (_try_torchvision, _try_download):
        try:
            imgs = fn(split, root)
            return imgs.astype(np.float32) / 255.0, "MNIST"
        except Exception:
            pass
    imgs = _sklearn_digits()
    # Without MNIST: first 1500 images for training, the rest for testing
    imgs = imgs[:1500] if split == "train" else imgs[1500:]
    return imgs.astype(np.float32) / 255.0, "sklearn-digits(28x28)"


MEDMNIST_2D = ["bloodmnist", "pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist",
               "retinamnist", "breastmnist", "tissuemnist", "organamnist", "organcmnist", "organsmnist"]
# Microscopy images are inverted by default: bright-field "bright background, dark cells"
# -> "dark background, bright cells", mimicking fluorescence images
_MICROSCOPY = {"bloodmnist", "pathmnist", "tissuemnist"}


def _load_medmnist(name, split, size, root):
    import medmnist
    from medmnist import INFO
    os.makedirs(root, exist_ok=True)
    cls = getattr(medmnist, INFO[name]["python_class"])
    try:
        ds = cls(split=split, download=True, root=root, size=size)
    except TypeError:                       # old medmnist (v2) has no size argument, 28x28 only
        if size != 28:
            raise RuntimeError("This medmnist version only supports 28x28; run: pip install -U medmnist")
        ds = cls(split=split, download=True, root=root)
    imgs = np.asarray(ds.imgs).astype(np.float32) / 255.0
    if imgs.ndim == 4:                      # RGB -> grayscale (a single-pixel detector measures total intensity)
        imgs = imgs @ np.array([0.299, 0.587, 0.114], np.float32)
    return imgs


def load_dataset(name="bloodmnist", split="train", size=28, invert=None, root="./data"):
    """Returns ((N,size,size) float32 images in [0,1], dataset description).
    name: 'mnist' or a MedMNIST 2D dataset name (default bloodmnist: blood-cell microscopy).
    invert: if None, microscopy datasets are inverted automatically."""
    name = name.lower()
    if name == "mnist":
        if size != 28:
            raise ValueError("mnist only supports size=28")
        return load_digits_28(split, root)
    if name not in MEDMNIST_2D:
        raise ValueError(f"Unknown dataset {name}. Options: mnist, {', '.join(MEDMNIST_2D)}")
    try:
        imgs = _load_medmnist(name, split, size, root)
    except ImportError:
        raise SystemExit("Please install medmnist first:  pip install medmnist")
    except RuntimeError as e:
        raise SystemExit(
            f"Automatic download of {name}.npz failed. Try one of:\n"
            "  1. pip install certifi, then run again (fixes the macOS certificate issue)\n"
            "  2. Run /Applications/Python\\ 3.10/Install\\ Certificates.command (match your Python version)\n"
            f"  3. Download https://zenodo.org/records/10519652/files/{name}.npz?download=1 in a browser "
            f"and put it in {os.path.abspath(root)}\nOriginal error: {str(e).strip()[:200]}")
    if invert is None:
        invert = name in _MICROSCOPY
    if invert:
        imgs = 1.0 - imgs
    return imgs, f"{name} {size}x{size}{' (inverted, fluorescence-style)' if invert else ''}"


def ssim_psnr(rec, gt):
    """rec/gt: (H,W) in [0,1]."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    return (structural_similarity(gt, rec, data_range=1.0),
            peak_signal_noise_ratio(gt, rec, data_range=1.0))


def minmax(img):
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-12)
