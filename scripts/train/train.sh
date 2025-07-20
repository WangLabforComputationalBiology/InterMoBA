###
# Energy
PYTHONPATH=interformer/ CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -data_path interformer/train/general_PL_2020.csv \
-work_path interformer/poses \
-ligand ligand/rcsb \
-seed 1111 \
-filter_type normal \
-native_sampler 0 \
-Code Energy \
-batch_size 2 \
-gpus 4 \
-method Gnina2 \
-patience 30 \
-early_stop_metric val_loss \
-early_stop_mode min \
-affinity_pre \
--warmup_updates 110000 \
--peak_lr 0.00012 \
--n_layers 6 \
--hidden_dim 128 \
--num_heads 8 \
--dropout_rate 0.1 \
--attention_dropout_rate 0.1 \
--weight_decay 1e-5 \
--energy_mode True