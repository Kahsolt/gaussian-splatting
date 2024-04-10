### Folder Layout

```
.
├── README.md
├── LICENSE.md
├── assets
├── environment.*       # conda venv
├── SIBR_viewers_bin    # SIBR viewer
├── submodules          # dependent git repos (e.g. dgr engines)
│   ├── diff-gaussian-rasterization
│   ├── diff-gaussian-rasterization-kahsolt
│   ├── simple-knn
│   └── init_repos.cmd
├── data                # dataset inputs
├── output              # result ouputs
├── modules
│   ├── data            # dataloaders
│   ├── lpipsPyTorch    # LPIPS metric
│   ├── hparam.py       # default hparams
│   ├── camera.py       # default camera
│   ├── scene.py        # default scene
│   ├── layers.py       # common neural network layers/blocks/modules
│   ├── utils           # common utils
│   ├── network_gui.py  # remote monitor for SIBR viewer
│   └── morphs          # implementing 3d-gs variations
│       ├── README.md
│       └── <morph>
│           ├── model.py
│           ├── render.py
│           ├── train.py
│           ├── hparams.py    # override default
│           ├── camera.py     # override default
│           ├── scene.py      # override default
│           └── ...
│── run.cmd             # example run commands
├── train.py            # train entry script
├── render.py           # render entry script
├── metrics.py          # metric entry script
├── convert.py          # dataset convert entry script (not used)
└── full_eval.py        # full eval entry script (not used)
```


### Class Layout

⚪ 引用关系

```
- Scene: 场景（顶级对象，一个实验，一切信息都能通过它取得）
  - HyperParams: 超参
  - Camera: 相机真值
  - GaussianModel: 可训练的高斯模型
```

⚪ 构造关系

```python
hp = HyperParams()
scene = Scene(hp)
scene.train_cameras: List[Camera]
scene.test_cameras: List[Camera]
scene.gaussians: GaussianModel
```
