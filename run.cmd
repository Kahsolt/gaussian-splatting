:default
python train.py -s data\tandt\train
python render.py -m output\e2e87895-f
python metrics.py -m output\e2e87895-f


:white_background
python train.py -s data\tandt\train -w
python render.py -m output\ac0e8170-5
