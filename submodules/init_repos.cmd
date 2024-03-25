@ECHO OFF
REM add more rasterizer versions

:ours
git clone https://github.com/Kahsolt/diff-gaussian-rasterization diff-gaussian-rasterization-ks

:others
git clone https://github.com/ingra14m/depth-diff-gaussian-rasterization depth-diff-gaussian-rasterization-ingra14m
git clone https://github.com/leo-frank/diff-gaussian-rasterization-depth
git clone https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth
git clone https://github.com/tobias-kirschstein/depth-diff-gaussian-rasterization
