:envvar
REM setup compiler envvar
CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
SET DISTUTILS_USE_SDK=1
REM add SIBR Viewer to path
PATH SIBR_viewers_bin\bin\;%PATH%

:viewers
REM monitor gs training process
SIBR_remoteGaussian_app.exe
REM render pretrained gs model
SIBR_gaussianViewer_app.exe -m output\original-train
REM render init point cloud
SIBR_PointBased_app.exe --path data\tandt\train
REM ?
SIBR_texturedMesh_app.exe --path data\tandt\train

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:default
REM run normal train & eval routine
python train.py -s data\tandt\train -m output\original-train
python render.py -m output\original-train
python metrics.py -m output\original-train

:morphs
REM specify your morphy
python train.py -M mlp_gs -s data\tandt\train -m output\mlp_gs-train
python train.py -M cd_gs -s data\tandt\train -m output\cd_gs-train
python train.py -M if_gs -s data\tandt\train -m output\if_gs-train
python train.py -M gs_w -s data\tandt\train -m output\gs_w-train

:vis
python vis.py -m output\mlp_gs
python vis_sun.py -m output\mlp_gs


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:loss_mask
python train.py -M dev --m_loss_weight
python train.py -M dev --m_loss_depth
python train.py -M dev --m_loss_depth_reverse
python train.py -M dev --m_loss_importance

:prune_by_importance
python train.py -M dev -m output\train-importance --m_loss_importance
python train.py -M dev -m output\train-importance --load_iter -1 --sanitize_load_gauss
SIBR_remoteGaussian_app.exe

:debug_render_set
python train.py -M mlp_gs -s data\phototourist\brandenburg_gate --eval
python render.py -m "output\2024-04-28T17-52-59_brandenburg_gate_M=mlp_gs" --debug_render_set --limit 30
