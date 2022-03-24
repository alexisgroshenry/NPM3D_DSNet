## Requirements
- easydict
- [hdbscan](https://pypi.org/project/hdbscan/)
- numba
- numpy
- pyyaml
- python=3.7
- scikit-learn
- scipy
- [spconv](https://github.com/traveller59/spconv)=1.1 or 1.2
- tensorboard=2.3.0
- torch=1.5
- torchvision=0.6.0
- [torch-cluster](https://github.com/rusty1s/pytorch_cluster)=1.5.7
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)=1.3.2
- tqdm

## Data Preparation

SemanticKITTI dataset should be downloaded and organized in the folder `data` as following :

```
./
├── 
├── ...
└── data/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	        └── ...
```

## Training

Use the scripts `./scripts/xxx/train_xxx.sh` to train the backbone (only semantic branch), or the dynamic-shifting module on the training sequences of the SemanticKITTI dataset. The scripts are available for pytorch and slurm.

## Evaluation

Use the scripts `./scripts/xxx/val_xxx.sh` to evaluate the backbone (only semantic branch), or the dynamic-shifting module on the sequence 08 of the SemanticKITTI dataset. The scripts are available for pytorch and slurm and compute several metrics of the computed results.

## Test

Use the scripts `./scripts/xxx/test_xxx.sh` to test the backbone (only semantic branch), or the dynamic-shifting module on the test sequences of the SemanticKITTI dataset (11 to 21). The scripts are available for pytorch and slurm.

## Results visualization

Run the code `./vis/visualize.py` with the wanted arguments to display an interactive visualization of the groundtruth and of the results. A description of the arguments can be accessed with option `--help`.