# PyInstaller hook for scipy
from PyInstaller.utils.hooks import collect_submodules

# Collect all scipy submodules
hiddenimports = collect_submodules('scipy')

# Additional imports
hiddenimports += [
    'scipy.signal',
    'scipy.io',
    'scipy.io.wavfile',
    'scipy.fft',
    'scipy.fftpack',
    'scipy.ndimage',
    'scipy.sparse',
    'scipy.linalg',
    'scipy.special',
    'scipy.stats',
]
