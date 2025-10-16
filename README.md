

# VLN-Zero
Rapid Exploration and Cache-Enabled Neurosymbolic Vision-Language Planning for Zero-Shot Transfer in Robot Navigation


![Framework Overview](https://vln-zero.github.io/static/images/vln-zero-framework.png)


## Installation

1. Clone this repo

```bash
cd VLN-Zero
```

2. Make changes to the `habitat-lab` code

In `habitat-lab/habitat/utils/visualizations/maps.py`, change lines 425-426 to the following:

```python
    limit = topdown_map_info["limits"]
    # Crop the map to show only the agent and goal +- some buffer
    top_down_map = top_down_map[limit[0]:limit[1], limit[2]:limit[3]]
```

Also, in `habitat-lab/habitat_baselines/rl/requirements.txt`, remove lines 3-4 (the `tensorflow==1.13.1` requirement is not needed).

3. Follow the VLN-CE installation guide.

Install both Habitat-Lab and VLN-CE following the setup steps provided [here](https://github.com/jacobkrantz/VLN-CE).

4. Download data

Following the steps from the VLN-CE project, download the MP3D, R2R, and RxR datasets. The final structure should look like something like this.

```
VLN-Zero
├─ habitat-lab
├─ VLN_CE
│  ├─ data
│  │  ├─ datasets
│  │  │  ├─ R2R_VLNCE_v1-3
│  │  │  │  ├─ test
│  │  │  │  ├─ train
│  │  │  │  ├─ val_seen
│  │  │  │  ├─ val_unseen
│  │  │  ├─ R2R_VLNCE_v1-3_preprocessed
│  │  │  │  ├─ envdrop
│  │  │  │  ├─ joint_train_envdrop
│  │  │  │  ├─ test
│  │  │  │  ├─ train
│  │  │  │  ├─ val_seen
│  │  │  │  ├─ val_unseen
│  │  │  ├─ RxR_VLNCE_v0
│  │  │  │  ├─ train
│  │  │  │  ├─ val_seen
│  │  │  │  ├─ val_unseen
│  │  │  │  ├─ test_challenge
│  │  │  │  ├─ text_features
│  │  ├─ ddppo-models
│  │  ├─ res
│  │  ├─ scene_datasets
│  │  │  ├─ mp3d
```

## Evaluation
To run our implementation, run ```bash eval_zero_vlnce.sh```. The number of gpus used can me modified in this script. Make sure to set the OPENAI_API_KEY variable to your key (Warning! With multiple gpus in use, this script will use a LOT of API calls).

Results can be tracked by running ```python analyze_results.py --path YOUR_PATH```

The evaluation can be killed by running ```bash kill_zero_eval.sh```

These scripts were taken from [NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE).


## Citation
Please cite with
```
@misc{anonymous2025vlnzerorapidexplorationcacheenabled,
      title={VLN-Zero: Rapid Exploration and Cache-Enabled Neurosymbolic Vision-Language Planning for Zero-Shot Transfer in Robot Navigation}, 
      author={Anonymous},
      year={2025},
      url={https://vln-zero.github.io/},
}
```
