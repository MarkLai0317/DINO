# environment
ubuntu 20.04
cuda 11.7
python 3.8.10

# install

1. create .venv environment
```bash
python -m venv .venv
source .venv/bin/activate
```
2. use pip to install required package
``` bash
pip install numpy==1.22.4
pip install wheel
pip install yapf==0.40.1
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install torchmetrics
```

3. compiling CUDA operators
```bash
cd models/dino/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

# put the data, back bones, and checkpoints in correct directories

1. put train, validation, test provided by hw1 in ./data directory and name them as below

```
/data
|--/annotations
|  |--instances_train2017.json
|  |--instances_val2017.json
|--/test
|--/train2017
|--/val2017

```
2. put  swin backbone and pretrain checkpoints in ./pretrain_model

```
/pretrain_model
|--checkpoint0029_4scale_swin.pth
|--swin_large_path4_window12_384_22k.pth
```


# re-generaate Output_json_for_test.json
3. open inference_and_visualization.ipynb and press run all and then it will generate Output_json_for_test.json if you want to recreate my result

# train model

1. open scripts/DINO_train_swin.sh

```bash
coco_path=$1
backbone_dir=$2
export CUDA_VISIBLE_DEVICES=$3 && python main.py \
	--output_dir logs/DINO/R50-MS4 -c config/DINO/DINO_4scale_swin.py --coco_path $coco_path \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 backbone_dir=$backbone_dir \
    --pretrain_model_path pretrain_model/checkpoint0029_4scale_swin.pth \
    --finetune_ignore label_enc.weight class_embed
```
replace output_dir logs with other name like test_logs/DINO/R50-MS4

2. run below command in this homework root directory
```
bash scripts/DINO_train_swin.sh ./data ./pretrain_model 0
```

# check the prediction of validation data. there are two method
1. run with data and checkpoint
```
bash scripts/DINO_eval.sh ./data [path to checkpoints]
```
example:

```
bash scripts/DINO_eval.sh ./data result_model/checkpoint0049.pth
```

2. run using evaluate.py
```
python evaluate.py ./val_result.json ./data/annotations/instances_val2017.json
```


# test my predict Output_json_for_test.json
```
python evaluate.py Output_json_for_test.json [test_truth_path]
``` 




