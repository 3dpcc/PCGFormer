# PCGFormer: Lossy Point Cloud Geometry Compression via Local Self-Attention

We propose the PCGFormer by embedding the local self-attention in the framework of multiscale sparse tensor representation, with which spatial neighbors can be effectively
aggregated and embedded for better compression performance. This local self-attention is implemented using the popular Transformer architecture where adaptive information aggregation 
is applied. As such, we dynamically weighted the contribution of local neighborhood through the use of kNN search for each individual point.
## News

- Our paper has been accepted by **VCIP2022**! [[paper](https://ieeexplore.ieee.org/abstract/document/10008892)]

## Requirments
- python3.7 or 3.8
- cuda10.2 or 11.0
- pytorch1.7 or 1.8
- MinkowskiEngine 0.5 or higher (for sparse convolution)
- torchac 0.9.3 (for arithmetic coding) https://github.com/fab-jul/torchac
- tmc3 v12 (for lossless compression of downsampled point cloud coordinates) https://github.com/MPEGGroup/mpeg-pcc-tmc13

We recommend you to follow https://github.com/NVIDIA/MinkowskiEngine to setup the environment for sparse convolution. 

- Pretrained Models: ./ckpts_pretrain/

## Usage

### Testing
Please download the pretrained models and install tmc3 mentioned above first.
```shell
sudo chmod 777 tmc3 pc_error_d
```
For rate point 01
```shell
python test.py --filedir='8iVFB' --res='1023' --ckptdir='./ckpt_pretrain/r01_0.05bpp/epoch_last.pth' --pct_pos
```
For rate point 02-r04
```shell
python test.py --filedir='8iVFB' --res='1023' --ckptdir='./ckpt_pretrain/r02_0.25bpp/epoch_last.pth'
python test.py --filedir='8iVFB' --res='1023' --ckptdir='./ckpt_pretrain/r03_0.5bpp/epoch_last.pth'
python test.py --filedir='8iVFB' --res='1023' --ckptdir='./ckpt_pretrain/r04_0.6bpp/epoch_last.pth'
```
The testing rusults can be found in `./results`

local kNN Transformer is in transformer_pos.py and transformer.py.

The difference is whether there is a position embedding, the use of position embbeding at low bit rate points has better performance.
It is recommended that you use transformer_pos.py.

## Authors
These files are provided by HangZhou Normal University and Nanjing University. 
Please contact us (liugexin@stu.hznu.edu.cn, DandanDing@hznu.edu.cn, and mazhan@nju.edu.cn) if you have any questions.
