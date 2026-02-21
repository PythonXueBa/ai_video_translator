# PyInstaller hook for demucs
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect demucs data files
datas = collect_data_files('demucs')

# Collect all demucs submodules
hiddenimports = collect_submodules('demucs')

# Additional imports
hiddenimports += [
    'demucs.model',
    'demucs.pretrained',
    'demucs.separate',
    'demucs.apply',
    'demucs.audio',
    'demucs.repo',
]
