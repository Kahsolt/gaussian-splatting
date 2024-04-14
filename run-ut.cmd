@ECHO OFF

SETLOCAL ENABLEDELAYEDEXPANSION

SET DEV_FLAGS=
SET DEV_OUTPUT=output
REM SET DEV_FLAGS=--iterations 50 --test_iterations 50 --save_iterations 50 --limit 10
REM SET DEV_OUTPUT=output-ut

GOTO start

:run
echo Running morph: %1
echo ^[Step 1/3^] train
python train.py -M %1 -m !DEV_OUTPUT!\train-%1 !DEV_FLAGS!
echo ^[Step 2/3^] render
python render.py -m !DEV_OUTPUT!\train-%1 !DEV_FLAGS!
echo ^[Step 3/3^] metrics
python metrics.py -m !DEV_OUTPUT!\train-%1
echo ============================================================
EXIT /B 0

:start
CALL :run gs
CALL :run mlp_gs
CALL :run cd_gs
CALL :run if_gs
CALL :run gs_w
CALL :run dev
