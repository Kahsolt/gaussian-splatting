### Folder layout

This folder contains various 3D-GS implementations, each sub-folder is a runnable entry.  
You should implement `model.py`, `train.py` and `render.py` by yourself, the `hparam.py` is optional and you will find the base class at `modules.hparam.HyperParams`  
Also, you could alway copy from the original 3D-GS implementation `morphs/gs`, and then modify on your own :)  

```
├── gs
│   ├── hparam.py
│   ├── model.py
│   ├── render.py
│   └── train.py
└── <morph>
    └── ...
```


### Evolution hierachy

- gs: original
  - mlp_gs: replace SH with color-mlp
    - cd_gs: rendered image decomposition
      - cd_add_gs: rendered = center (视角无关的基本色/均色) + shift (视角独立的明暗偏差)
      - cd_mul_gs: rendered = rgb (~=视角无关的基本色/均色) * gate (视角独立的值域放缩)
        - cd_mulx_gs: rendered = rgb (~=视角无关的基本色/均色) * gate (视角独立的值域放缩，使用独立特征)
    - if_gs: multi-freq gaussian
    - gs_w: add gaussian importance, appearance & occlusion embedding
      - fgbg_gs: rendered = where(mask > 0, fg, bg)
    - occlu_gs: absorb occlusions in gt
  - dev: free experimental playground (still use SH)
