@ECHO OFF
REM add more rasterizer versions

SETLOCAL ENABLEDELAYEDEXPANSION

SET GCR=git clone --recursive
SET GC=git clone


:original
%GCR% https://github.com/graphdeco-inria/diff-gaussian-rasterization

:ours
%GC% https://gitee.com/kahsolt/diff-gaussian-rasterization diff-gaussian-rasterization-kahsolt


:others
REM detached, add depth
%GC% https://github.com/leo-frank/diff-gaussian-rasterization-depth

REM add depth & alpha
%GC% https://github.com/ashawkey/diff-gaussian-rasterization diff-gaussian-rasterization-ashawkey

REM add depth, 4th-degree SH
%GC% https://github.com/ingra14m/depth-diff-gaussian-rasterization depth-diff-gaussian-rasterization-ingra14m
REM fork of the ingra14m, add package prefix "depth_" to avoid name conflicting
%GC% https://github.com/tobias-kirschstein/depth-diff-gaussian-rasterization

REM add fwd for "median depth"
%GC% https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth

REM add depth, SE3 pose
%GC% https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose

REM add depth, bwd for pose; SLAM use
%GC% https://github.com/NPU-YanChi/diff-gaussian-rasterization-for-gsslam

REM add depth, alpga, objectId; semantic use
%GC% https://github.com/fusheng-ji/diff-gaussian-rasterization-modified

REM add accuracy map
%GC% https://github.com/robot0321/diff-gaussian-rasterization-depth-acc

REM add gauss confidence
%GC% https://github.com/zehaozhu/diff-gaussian-rasterization-confidence

REM use e3nn fmt for SH
%GC% https://github.com/dcharatan/diff-gaussian-rasterization-modified diff-gaussian-rasterization-modified-dcharatan

REM light-weight
%GC% https://github.com/Kevin-2017/compress-diff-gaussian-rasterization


:link_glm
REM for all diff-gaussian-rasterization derived repos, share the same thirdparty/glm with softlink :)
FOR /F %%f IN ('DIR /O:NG /T:W /B /AD .') DO (
  REM ECHO ^>^> Enter workdir %%f
  PUSHD %%f
  DIR /A /B third_party\glm | FINDSTR .* >NUL
  IF ERRORLEVEL 1 (
    RMDIR third_party\glm
    MKLINK /J third_party\glm ..\diff-gaussian-rasterization\third_party\glm
  )
  REM ECHO ^<^< Leave workdir %%f
  POPD
)
