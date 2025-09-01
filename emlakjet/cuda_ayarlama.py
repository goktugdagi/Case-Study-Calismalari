# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"using device: {device}")

# check_cuda.py
import torch
print("torch:", torch.__version__)
print("is_cuda_available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("cudnn:", torch.backends.cudnn.version())

#########################################################3
# PYG kurulumu
# install_pyg.py
import subprocess
import sys
import torch

# Torch sürümünü ve CUDA bilgisini al
tv = torch.__version__.split('+')[0]  # örn: '2.4.0'
cudatag = 'cpu' if torch.version.cuda is None else 'cu' + torch.version.cuda.replace('.', '')
url = f"https://data.pyg.org/whl/torch-{tv}+{cudatag}.html"

print("[INFO] Using:", url)

# PyG paketlerini kur
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric",
    "-f", url
])

print("[OK] PyG installed.")
