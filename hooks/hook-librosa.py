# PyInstaller hook for librosa
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect librosa data files
datas = collect_data_files('librosa')

# Collect all librosa submodules
hiddenimports = collect_submodules('librosa')

# Additional imports
hiddenimports += [
    'librosa.core',
    'librosa.feature',
    'librosa.util',
    'librosa.display',
    'librosa.effects',
    'librosa.filters',
    'librosa.onset',
    'librosa.segment',
    'librosa.sequence',
    'librosa.beat',
    'librosa.decompose',
    'sklearn',
    'sklearn.decomposition',
    'sklearn.cluster',
    'sklearn.feature_extraction',
]
