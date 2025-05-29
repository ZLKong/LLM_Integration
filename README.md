# Enabling Flexible Multi-LLM Integration for Scalable Knowledge Aggregation


<p align="center">
  <img src="./framework_fusionx.png" width=60%> <br>
</p>



## 💻 Usage

### Environment Settings

```
conda create -n fusionx python=3.9

conda activate fusionx

pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```
### 📚 Data Processing
Please follow the detailed guidelines in https://github.com/fanqiwan/FuseAI 

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
