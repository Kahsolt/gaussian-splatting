REM follow this script to build a toolchain of
:: python 3.9 + CUDA 12 + torch 2.x

REM install CUDA 12
:: https://developer.nvidia.com/cuda-toolkit
REM install VS 20xx components: C++ Program Tools & Universal CRT SDK
:: https://visualstudio.microsoft.com/zh-hans/downloads/?q=build+tools

conda create -y -n gs python==3.9
conda activate gs
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tqdm pyreadline
pip install plyfile pyfftw

REM open VS 20xx x64 Native Command Prompt
SET DISTUTILS_USE_SDK=1
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization-kahsolt
pip install submodules/simple-knn
pip install submodules/pytorch_wavelets
