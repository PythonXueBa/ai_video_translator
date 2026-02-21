# PyInstaller hook for soundfile
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect soundfile libraries
binaries = collect_dynamic_libs('soundfile')

# Collect data files
datas = collect_data_files('soundfile')
