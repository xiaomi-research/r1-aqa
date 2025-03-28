# R1-AQA --- Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering

## Introduction

R1-AQA is a audio question answering (AQA) model based on `Qwen2-Audio-7B-Instruct`, optimized through reinforcement learning (RL) using the group relative policy optimization (GRPO) algorithm.
This implementation has achieved state-of-the-art performance on the MMAU benchmark with only 38k post-training samples.

Our main findings are as follows:

- The GRPO algorithm can be directly and effectively applied to the audio modality, even to `Qwen2-Audio-7B-Instruct` with only 8.2B parameters.
- With only 38k post-training samples, reinforcement learning outperforms supervised fine-tuning, indicating that RL-based approaches can be effective without large datasets.
- The explicit reasoning process has not shown significant benefits for AQA tasks, and how to efficiently leverage *deep thinking* or step-by-step reasoning remains an open question for further research.
- Large audio language models (LALMs) still lag far behind humans auditory-language reasoning, suggesting that the RL-based approaches warrant further explorations.

Additional Notes:  

- The AVQA training set originally consists of approximately 40k samples. However, we use only about 38k samples because some data sources have become invalid. Other datasets using YouTube sources face a similar issue, such as AudioSet. We believe that the missing 2k samples do not have a significant impact on the training results.
- The statement about the 8.2B parameters is based on the *Qwen2-Audio Technical Report*.

### Table: Accuracies (%) on the MMAU benchmark

| Model                                 | Method                | Test-mini | Test  | Test-mini | Test  | Test-mini | Test  | Test-mini | Test  |
|---------------------------------------|-----------------------|-----------|-------|-----------|-------|-----------|------|------------|-------|
| -                                     | Human\*               | 86.31     | -     | 78.22     | -     | 82.17     | -     | 82.23     | -     |
| Gemini Pro 2.0 Flash                  | Direct Inference\*    | 56.46     | 61.73 | 58.68     | 56.53 | 51.65     | 61.53 | 55.60     | 59.93 |
| Audio Flamingo 2                      | Direct Inference\*    | 61.56     | 65.10 | 73.95     | 72.90 | 30.93     | 40.26 | 55.48     | 59.42 |
| GPT4o + Strong Cap.                   | Direct Inference\*    | 57.35     | 55.83 | 49.70     | 51.73 | 64.86     | 68.66 | 57.30     | 58.74 |
| Llama-3-8B-Instruct + Strong Cap.     | Direct Inference\*    | 50.75     | 49.10 | 48.93     | 48.93 | 55.25     | 62.70 | 52.10     | 53.57 |
| Qwen2-Audio-7B-Instruct               | Direct Inference\*    | 54.95     | 45.90 | 50.98     | 53.26 | 42.04     | 45.90 | 49.20     | 52.50 |
| SALAMONN                              | Direct Inference\*    | 41.00     | 40.30 | 34.80     | 33.76 | 25.50     | 24.24 | 33.70     | 32.77 |
| Qwen2-Audio-7B-Instruct               | CoTA \[1\]            | 60.06     | -     | 64.30     | -     | 60.70     | -     | 61.71     | -     |
| Qwen2-Audio-7B-Instruct               | Zero-Shot-CoT \[2\]   | 61.86     | -     | 56.29     | -     | 55.26     | -     | 57.80     | -     |
| **Qwen2-Audio-7B-Instruct**           | **GRPO (Ours) 1️⃣**    | 69.37     | -     | 66.77     | -     | 57.36     | -     | 64.50     | -     |
| **Qwen2-Audio-7B-Instruct**           | **GRPO (Ours) 2️⃣**    | 68.77     | 69.76 | 64.37     | 61.40 | 63.66     | 62.70 | 65.60     | 64.36 |

#### Notes

\* The data are sourced from the [MMAU leaderboard](https://sakshi113.github.io/mmau_homepage/#leaderboard).  
\[1\] Xie, Zhifei, et al. "Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models." arXiv preprint arXiv:2503.02318 (2025).  
\[2\] Ma, Ziyang, et al. "Audio-CoT: Exploring Chain-of-Thought Reasoning in Large Audio Language Model." arXiv preprint arXiv:2501.07246 (2025).  
1️⃣ It is the original model, identical to the one on Hugging Face and described in our technical report.  
2️⃣ It is the model submitted to the [MMAU leaderboard](https://sakshi113.github.io/mmau_homepage/#leaderboard), trained multiple times to achieve balanced results.  

**Hugging Face:**  
[🤗 R1-AQA Model: mispeech/r1-aqa](https://huggingface.co/mispeech/r1-aqa)  

**arXiv:**  
[📝 Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering](https://arxiv.org/abs/2503.11197)

**R1-AQA Team:**  
[Gang Li](https://github.com/GrantL10)`*` · [Jizhong Liu](https://github.com/frankenliu)`*` · [Heinrich Dinkel](https://github.com/RicherMans) · [Yadong Niu](https://github.com/nyd3001) · [Junbo Zhang](https://github.com/jimbozhang) · [Jian Luan](https://github.com/jianluan)

`*` Equal contribution.

### Updates

- 2025-03-28: Update our results on the MMAU leaderboard.
- 2025-03-18: Support the mode containing `<think> </think>` (*GRPO + Prompt <3>* in our technical report).
- 2025-03-17: Release the R1-AQA repository.

## Training

### Data Preparation

We use the [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/) `training` subset (train_qa.json), and convert the data to the R1-AQA format, where each line in the text file represents a JSON object with specific keys

```json
{
    # The data presented below originate from the original AVQA dataset.
    "id": 183,
    "video_name": "-HG3Omg_89c_000030",
    "video_id": 341,
    "question_text": "What happened in the video?",
    "multi_choice": [  
        "motorboat",  
        "Yacht consignment",  
        "Sailboat set sail",  
        "Consignment car"  
    ],
    "answer": 1,
    "question_relation": "View",
    "question_type": "Happening", 
    # We add the following data.
    "dataset_name": "AVQA",
    "audio_path": "Path to wav dir/-HG3Omg_89c_30.wav"
}
```

### GRPO

```bash
sh run_grpo.sh
```

#### Wandb

![Image](./resources/wandb.png)

#### NOTE

- Replace the `DATA_FILE` variable in the `run_grpo.sh` with your dataset path.
- If you already have the `Qwen2-Audio-7B-Instruct` model, please modify the `MODEL_NP` variable in `run_grpo.sh` to your local model path.
- We recommend using a GPU with 80 GB of memory.

## Testing

### MMAU Test-mini

Evaluate the MMAU `Test-mini` dataset, please follow these steps:

- Download Data
  - To test the MMAU Test-mini dataset requires the following files from the [MMAU](https://github.com/Sakshi113/MMAU/tree/main) repository: [mmau-test-mini.json](https://github.com/Sakshi113/MMAU/blob/main/mmau-test-mini.json), [evaluation.py](https://github.com/Sakshi113/MMAU/blob/main/evaluation.py), and [test-mini-audios.tar.gz](https://drive.google.com/file/d/1fERNIyTa0HWry6iIG1X-1ACPlUlhlRWA/view?usp=sharing). The method for obtaining data is as follows:

```bash
mkdir -p data && cd data

git clone https://github.com/Sakshi113/MMAU.git

cd data/MMAU

# ***Check if output_key = 'model_output' in evaluation.py, change it to output_key = 'model_prediction'.***

#TODO you should download test-mini-audios.tar.gz to here
***download test-mini-audios.tar.gz to here***

# Uncompress wav files
tar -xzvf test-mini-audios.tar.gz

cd ../../
```

- Evaluation

```bash
# Testing MMAU test-mini with in every 100 steps. 
# You can uncomment the line 12 of test_mmau.sh to eval the entire MMAU, if you have downloaded test-audios.tar.gz.
# You can modify the script to test other steps or change other parameters.
sh test_mmau.sh
```

## Hacking It

We encourage hacking it on your own. If you want to see the "thinking" or improve our work, here are some hints:

> 1. Uncomment the line 25 of `src\dataset\dataset.py`;
> 2. Uncomment the line 55 of `src\utils\rewards.py`;
> 3. Uncomment the line 46 of `src\test.py`;
> 4. Train and test your model;
> 5. ***Design your CoT strategy based on `<think> </think>`. Let's explore effective ways to combine RL and CoT!***

## Acknowledgement

> 1. We have referred to the implementation of [R1-V](https://github.com/Deep-Agent/R1-V) for the GRPO-based training.
> 2. We sincerely thank [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/) and [MMAU](https://github.com/Sakshi113/MMAU/tree/main) for providing the datasets.

## Citation

```bib
@article{li2025reinforcement,
  title={Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering},
  author={Li, Gang and Liu, Jizhong and Dinkel, Heinrich and Niu, Yadong and Zhang, Junbo and Luan, Jian},
  journal={arXiv preprint arXiv:2503.11197},
  year={2025},
  url={https://github.com/xiaomi-research/r1-aqa; https://huggingface.co/mispeech/r1-aqa}
}
```
