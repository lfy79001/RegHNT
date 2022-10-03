# RegHNT

**Re**lational **G**raph enhanced **H**ybrid table-text **N**umerical reasoning model with **T**ree decoder.

This is the project containing source code for the paper [Answering Numerical Reasoning Questions in Table-Text Hybrid Contents with Graph-based Encoder and Tree-based Decoder](https://arxiv.org/abs/2209.07692) in __COLING 2022__. 

__Please kindly cite our work if you find our codes useful, thank you.__
```bash
@article{lei2022answering,
  title={Answering Numerical Reasoning Questions in Table-Text Hybrid Contents with Graph-based Encoder and Tree-based Decoder},
  author={Lei, Fangyu and He, Shizhu and Li, Xiang and Zhao, Jun and Liu, Kang},
  journal={arXiv preprint arXiv:2209.07692},
  year={2022}
}
```



## requirements

To create an environment with `conda` and activate it.

```bash
conda create -n reghnt python==3.7
conda activate reghnt
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html     # Adjust according to your CUDA version
pip install allennlp==0.8.4 transformers==4.21.1 nltk==3.5 pandas==1.1.5 numpy==1.21.6
conda install -c dglteam dgl-cuda11.1==0.6.1    # Adjust according your CUDA version
pip install sentencepiece
```
Next, you should install `torch-scatter==2.0.5` (python3.7 CUDA11.1) by
__Download [torch-scatter wheel](https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_scatter-2.0.5-cp37-cp37m-win_amd64.whl).__ (already existed)

__Or download [other vision](https://pytorch-geometric.com/whl/) according to your Python version and CUDA version. Then move it to `RegHNT/`.__
```
pip install torch_scatter-2.0.5-cp37-cp37m-linux_x86_64.whl
```

We adopt `RoBERTa` as our encoder to develop our RegHNT and use the following commands to prepare RoBERTa model
```bash
cd dataset_reghnt
mkdir roberta.large && cd roberta.large
wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin
wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json
wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json
wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt
```

## Training

### Preprocessing dataset
We use the preprocessed data by [TagOp Model](https://github.com/NExTplusplus/tat-qa) and they are already in this repository.

### Prepare dataset

```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/reg_hnt python reg_hnt/prepare_dataset.py --mode [train/dev/test]
```

Note: The result will be written into the folder `./reg_hnt/cache` default.

### Train
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python reg_hnt/trainer.py --data_dir reg_hnt/cache/ \
--save_dir ./try --batch_size 48 --eval_batch_size 1 --max_epoch 100 --warmup 0.06 --optimizer adam --learning_rate 1e-4 \
--weight_decay 5e-5 --seed 42 --gradient_accumulation_steps 12 --bert_learning_rate 1e-5 --bert_weight_decay 0.01 \
--log_per_updates 50 --eps 1e-6 --encoder roberta_large --roberta_model dataset_reghnt/roberta.large
```

## Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python reg_hnt/predictor.py --data_dir reg_hnt/cache/ \
--test_data_dir reg_hnt/cache/ --save_dir reg_hnt --eval_batch_size 1 --model_path ./try \
--encoder roberta_large --roberta_model dataset_reghnt/roberta.large --mode dev
```
```
python tatqa_eval.py --gold_path=dataset_reghnt/tatqa_dataset_dev.json --pred_path=reg_hnt/pred_result_on_dev.json
```

## Testing
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python reg_hnt/predictor.py \
--data_dir reg_hnt/cache/ --test_data_dir reg_hnt/cache/ --save_dir reg_hnt \
--eval_batch_size 1 --model_path ./try --encoder roberta_large --roberta_model dataset_reghnt/roberta.large --mode test
```

Note: The training process may take around 3 days using a single 24GB RTX3090.


## Any Question?

For any issues please create an issue [here](https://github.com/lfy79001/RegHNT/issues) or kindly email us at:
Fangyu Lei [843265183@qq.com](mailto:843265183@qq.com)
