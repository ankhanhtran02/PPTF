
# Continual Learning for Seq2seq models for Code related tasks.

This project contains the code for Continual Learning for Seq2seq models for Code related tasks.


#### Relevant Repositories
This codebase has some code or ideas ported from the following repositories.
1. [CodeXGLUE](https://github.com/microsoft/CodeXGLUE)
2. [CodeT5](https://github.com/salesforce/CodeT5)
3. [Learning to Prompt](https://github.com/google-research/l2p)

#### Folder Structure
```
src
│   dataloaders	# dataloader for CL tasks.
│   evaluator	# Original from CodeT5 for evaluation.
│	plots	# File to make some basic plots related to similarity and prompt frequency.
|	tokenizer	# Original from CodeT5 for some tokenization. Not used for us.
└───sh
|	|	final_runs.sh	# Contains commands for main experiments. These can be used as examples to run the code. For more info on the arguments please look at the config.py file.
└───models
|	|	T5prompt.py
└───utils
│   │   metrics.py	# Main file which implements the metrics.
│   │   replay.py	# Main file to implement replay buffer.
│   │	configs.py	# argparse arguments, etc
│   cont_gen.py	# Main file for running CL experiments.
|	analyse.ipynb	# Main file for analysing the query-key matching analysis.
|	run_gen.py	# Original file from codeT5 to finetune on a single file.
|	run_multi_gen.py	# Modified file from CodeT5 to run multitask learning. Has some hacks to get it works for us.
```

#### Usage
1. Set up environment:
```
conda create -n venv python=3.10.11
conda activate venv
cd src
pip install -r requirements.txt
export WANDB_API_KEY="[YOUR API KEY]"
```

2. Run command for PP + TF:
```
python cont_gen.py --pool_teacher=train/both --num_shared_keys_per_pair=2 --keys_agg=random --pool_freq --name=pool100_teacher --pool_size=100 --prompt_method=pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=16 --eval_batch_size=64 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=teacher_tune --stream=concode_none,translate_java-cs,summarize_ruby,refine_small --project_dir=experiment
```
