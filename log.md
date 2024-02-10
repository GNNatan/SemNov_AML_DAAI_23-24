# OOD Detection in Point Clouds
```python
import torch
print(torch.__version__)
```
```python
# installing pointnet++
!pip install ninja
!git clone https://github.com/erikwijmans/Pointnet2_PyTorch
%cd /content/Pointnet2_PyTorch
!python setup.py install

# moving to root directory
%cd ..
```
```python
!pip install "https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl"
```
```python
# install minimal requirements (pytorch is already installed in colab)
!pip install timm==0.5.4 wandb tqdm h5py==3.6.0 protobuf==3.20.1 lmdb==1.2.1 msgpack-numpy==0.4.7.1 scikit-learn
```
```python
# fork of the teacher's repo
!git clone https://github.com/GNNatan/SemNov_AML_DAAI_23-24.git

# Move to the project directory after Git clone
%cd /content/SemNov_AML_DAAI_23-24
```
```python
!sh download_data.sh
```
### Training 
```python
# Train DGCNN on SR1
#!python classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR1 --src SR1 --loss CE --wandb_proj AML_DAAI_proj23_24 --wandb_group sng-am
```

```python
# Train DGCNN on SR2
!python classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR2 --src SR2 --loss CE --wandb_proj AML_DAAI_proj23_24 --wandb_group sng-am
```

_For PointNet, we used a batch size of 16 due to memory concerns, using the default value required more memory than what is granted by Google Colabs and thus, was unfeasible to us._
```python
# Train PointNet2 on SR1
!python classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PointNet2_CE_SR1 --src SR1 --batch_size 16 --loss CE_ls --wandb_proj AML_DAAI_proj23_24 --wandb_group sng-am
```

```python
# Train PointNet2 on SR2
!python classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PointNet2_CE_SR2 --src SR2 --batch_size 16 --loss CE_ls --wandb_proj AML_DAAI_proj23_24 --wandb_group sng-am
```

### Testing
```python
!python classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR1_eval --src SR1 --loss CE -mode eval --ckpt_path outputs/DGCNN_CE_SR1/models/model_last.pth
```

```python
!python classifiers/trainer_cla_md.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SR2_eval --src SR2 --loss CE -mode eval --ckpt_path outputs/DGCNN_CE_SR2/models/model_last.pth
```

```python
!python classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PointNet2_CE_SR1_eval --src SR1 --loss CE -mode eval --ckpt_path outputs/PointNet2_CE_SR1/models/model_last.pth
```

```python
!python classifiers/trainer_cla_md.py --config cfgs/pn2-msg.yaml --exp_name PointNet2_CE_SR2_eval --src SR2 --loss CE -mode eval --ckpt_path outputs/PointNet2_CE_SR2/models/model_last.pth
```
