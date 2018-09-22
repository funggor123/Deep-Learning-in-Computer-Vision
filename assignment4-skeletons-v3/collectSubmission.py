import os
os.system("del assignment4.7z")
os.system("7z.exe a -r assignment4 . -xr!*cs231n/datasets* -xr!*ipynb_checkpoints* -xr!*README.md -xr!*collectSubmission* -xr!*requirements.txt -xr!*git* -xr!7z.exe")