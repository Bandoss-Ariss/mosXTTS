CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py \
--output_path=checkpoints/ \
--train_csv_path=dataset/metadata.csv \
--eval_csv_path=dataset/metadata_val.csv \
--language="mos" \
--num_epochs=10 \
--batch_size=64 \
--lr=5e-6 \