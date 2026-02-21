# PyInstaller hook for transformers
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect transformers data files
datas = collect_data_files('transformers')

# Collect all transformers submodules
hiddenimports = collect_submodules('transformers')

# Additional model-specific imports
hiddenimports += [
    'transformers.models',
    'transformers.models.m2m100',
    'transformers.models.m2m100.modeling_m2m100',
    'transformers.models.m2m100.tokenization_m2m100',
    'transformers.models.m2m100.configuration_m2m100',
    'transformers.models.whisper',
    'transformers.models.whisper.modeling_whisper',
    'transformers.models.whisper.tokenization_whisper',
    'transformers.models.whisper.configuration_whisper',
    'transformers.generation',
    'transformers.generation.utils',
    'transformers.tokenization_utils',
    'transformers.tokenization_utils_base',
    'transformers.modeling_utils',
]
