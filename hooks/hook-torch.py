# PyInstaller hook for torch
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

# Collect torch libraries
binaries = collect_dynamic_libs('torch')

# Collect torch data files
datas = collect_data_files('torch')

# Collect all torch submodules
hiddenimports = collect_submodules('torch')

# Additional hidden imports for torch
hiddenimports += [
    'torch.jit',
    'torch.jit._script',
    'torch.jit._trace',
    'torch.nn',
    'torch.nn.functional',
    'torch.utils',
    'torch.utils.data',
    'torchaudio',
    'torchaudio.transforms',
    'torchvision',
]
