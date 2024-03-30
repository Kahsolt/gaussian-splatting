:envvar
REM setup compiler envvar
CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
SET DISTUTILS_USE_SDK=1
REM add SIBR Viewer to path
PATH SIBR_viewers_bin\bin\;%PATH%


:default
python train.py -s data\tandt\train
python render.py -m output\e2e87895-f
python metrics.py -m output\e2e87895-f

:white_background
python train.py -s data\tandt\train -w
python render.py -m output\ac0e8170-5

:no_point_move
python train.py -s data\tandt\train --position_lr_init 0.0 --position_lr_final 0.0

:loss_mask
python train.py --m_loss_weight
python train.py --m_loss_depth
python train.py --m_loss_depth_reverse
python train.py --m_loss_importance

:resume
python train.py --m_loss_importance


:viewers
REM monitor gs training process
SIBR_remoteGaussian_app.exe
REM render pretrained gs model
SIBR_gaussianViewer_app.exe -m output\e2e87895-f
REM render init point cloud
SIBR_PointBased_app.exe --path data\tandt\train
REM ?
SIBR_texturedMesh_app.exe --path data\tandt\train
