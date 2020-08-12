# Shayari Generation Model

This repository contains code to generate Hindi Shayaris. Please see this Medium article for more details.

### Environment Setup
- Make sure CUDA >= 10.0 is installed to enable GPU acceleration.
- Clone this repository and do `pip install -r requirements.txt`
- Download pre-trained models from [here](https://www.dropbox.com/sh/7x7kdvv8d93dik3/AAAPckfqfBfjLl3P6429SJSMa?dl=0).

### Usage
```
python main.py [-h] --model_path MODEL_PATH --tokenizer_path TOKENIZER_PATH
              [--length LENGTH] [--num_samples NUM_SAMPLES]
              [--temperature TEMPERATURE] [--top_k TOP_K] [--top_p TOP_P]
              [--no_cuda] [--seed SEED]

Required arguments:
  --model_path MODEL_PATH
                        Path to pre-trained model
  --tokenizer_path TOKENIZER_PATH
                        Path to pre-trained tokenizer

Optional Arguments:
  -h, --help            Show this help message and exit
  --length LENGTH       Maximum length of sample (default: 100)
  --num_samples NUM_SAMPLES
                        Number of samples to generate for a prompt (default:
                        1)
  --temperature TEMPERATURE
                        Softmax temperature, temperature=0 implies greedy
                        sampling (default: 1.0)
  --top_k TOP_K         Value of k for top-k sampling, for k=0 top-k sampling
                        is not used (default: 0)
  --top_p TOP_P         Value of p for nucleus sampling (default: 0.9)
  --no_cuda             Avoid using CUDA when available (default: False)
  --seed SEED           Random seed for initialization (default: 2020)
```

For example:
```
python main.py --model_path models/mle-model/ --tokenizer_path tokenizer/ --top_p 0.8 --temperature 0.9
```