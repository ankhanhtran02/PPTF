# Command for PP + TF (pooling during training):

python cont_gen.py --pool_teacher=train --num_shared_keys_per_pair=2 --keys_agg=random --pool_freq --name=pool100_teacher --pool_size=100 --prompt_method=pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=16 --eval_batch_size=64 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=teacher_tune --stream=concode_none,translate_java-cs,summarize_ruby,refine_small --project_dir=pp_tf_pool-train

# Command for PP + TF (pooling during both phases):
python cont_gen.py --pool_teacher=both --num_shared_keys_per_pair=2 --keys_agg=random --pool_freq --name=pool100_teacher --pool_size=100 --prompt_method=pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=16 --eval_batch_size=64 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=teacher_tune --stream=concode_none,translate_java-cs,summarize_ruby,refine_small --project_dir=pp_tf_pool-both

# Command for PP + TF + ER:

python cont_gen.py --pool_teacher=train --num_shared_keys_per_pair=2 --replay=ring --buffer_size=500 --buffer_bs=2 --keys_agg=random --pool_freq --name=pool100_teacher_ER500 --pool_size=100 --prompt_method=pool --num_prompts_per_task=20 --train_only_prompts --bleu_samples=5000 --warmup_steps=500 --train_batch_size=16 --eval_batch_size=64 --log_steps=10 --data_num=-1 --save_last_checkpoints --always_save_model --project_name=teacher_tune --stream=concode_none,translate_java-cs,summarize_ruby,refine_small --project_dir=pp_tf_pool-train_ER500
