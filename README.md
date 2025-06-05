<p align="center">
  <img src="./framework_fusionx.png" width=60%> <br>
</p>


# Enabling Flexible Multi-LLM Integration for Scalable Knowledge Aggregation

[arXiv](https://arxiv.org/pdf/2505.23844) 

Existing methods for merging LLMs are memory-intensive and prone to task interference. We propose a framework that adaptively selects and fuses knowledge from multiple LLMs, enabling more scalable, stable, and memory-efficient integration—reducing interference by up to 50% compared to prior approaches.


## 💻 Usage

### Environment Settings

```
conda create -n fusionx python=3.9

conda activate fusionx

pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```
### 📚 Data Processing
Please follow the detailed guidelines in [FuseLLM](https://github.com/fanqiwan/FuseAI/tree/main/FuseLLM)

### ⭐ Training

```
sbatch train_deepspeed_local_dynamic.script
```


## 📝 Evaluation

### Commonsense Benchmark

```
git clone https://github.com/EleutherAI/lm-evaluation-harness.git

cd lm-evaluation-harness

pip3 install -e .

pip3 install omegaconf pycountry sentencepiece protobuf
```
#### Run Evaluation
```
bash run-lm-eval.sh
```

### BBH & MMLU Benchmark
```
git clone https://github.com/allenai/open-instruct.git

cd open-instruct

pip install -r requirements.txt
```

#### Run Evaluation
```
./scripts/data/prepare_eval_data.sh

bash scripts/eval/bbh.sh
```

## Citation
If you find our work useful in your research, please consider citing:
```
@article{kong2025enabling,
  title={Enabling Flexible Multi-LLM Integration for Scalable Knowledge Aggregation},
  author={Kong, Zhenglun and Zhan, Zheng and Hou, Shiyue and Gong, Yifan and Meng, Xin and Sui, Pengwei and Dong, Peiyan and Shen, Xuan and Wang, Zifeng and Zhao, Pu and others},
  journal={arXiv preprint arXiv:2505.23844},
  year={2025}
}
```
