
### Requirements
> - Python 3.8, PyTorch >= 1.11
> - BasicSR 1.4.2


### Installation
```
# Install dependent packages
cd MSLR
pip install -r requirements.txt
# Install BasicSR
python setup.py develop
```

### Training
Run the following commands for training:
```
# train MSLR for x2 effieicnt SR
python basicsr/train.py -opt options/train/MSLR/train_MSLR_X2.yml
```
- The train results will be in './experiments'.

### Testing
Run the following commands for testing:
```
# test MSLR for x2 efficient SR
python basicsr/test.py -opt options/test/MSLR/test_MSLR_x2.yml
```
- The test results will be in './results'.

