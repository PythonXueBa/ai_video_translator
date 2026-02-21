# PyInstaller hook for whisper
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect whisper model files
datas = collect_data_files('whisper')

# Collect all whisper submodules
hiddenimports = collect_submodules('whisper')
