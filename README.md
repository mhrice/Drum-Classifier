# Simple Drum Classifier

Quick and dirty drum classifier using a CNN. Classifies between kicks, snares, and hats.

Model: [AudioSet Tagging CNN](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py)

Dataset: [Drum and Percussion Kits](https://zenodo.org/record/3994999)
* Used freeDB.tar

Note: assumes you have a GPU for training, if not, remove `accelerator='gpu'`  in `train.py`. 

## Install
`pip install -r requirements.txt`
## Usage
### Training
`python train.py`

### Inference (requires best.ckpt)
`python inference.py path/to/audio/file/or/directory`

