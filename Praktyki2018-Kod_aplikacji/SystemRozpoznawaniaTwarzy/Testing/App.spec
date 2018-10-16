# -*- mode: python -*-

block_cipher = None


a = Analysis(['SystemRozpoznawaniaTwarzy.py'],
             pathex=['E:\\Repositories\\venv_face_recognition\\Lib\\site-packages\\scipy\\extra-dll', 'E:\\Repositories\\venv_face_recognition\\Lib\\site-packages\\PyQt5\\Qt\\bin', 'E:\\Repositories\\FaceRecognition\\Application', 'C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\Common7\\IDE\\Remote Debugger\\x64', 'E:\\Windows Kits\\10\\Redist\\ucrt\\DLLs\\x64'],
             binaries=[],
             datas=[],
             hiddenimports=['sklearn.neighbors.typedefs', 'PyQt5.sip', 'scipy._lib.messagestream', 'sklearn.tree', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='SystemRozpoznawaniaTwarzy',
          debug=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='SystemRozpoznawaniaTwarzy')