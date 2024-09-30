block_cipher = None

from PyInstaller.utils.hooks import collect_data_files
import ultralytics
ultralytics_files = collect_data_files("ultralytics")

a = Analysis(
    ['autocrop.py'],
    pathex=[],
    binaries=[
        ("ffmpeg.exe", "."),
        # ("ffprobe.exe", ".")
    ],
    # datas=[("lufsplot_icon.png", "lufsplot_icon.png")],
    datas=ultralytics_files,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='autocrop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="lufsplot_icon.png",
    version="autocrop_version_info.txt",
)
