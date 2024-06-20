CUDA_VISIBLE_DEVICES=2,3 python CT_MAR_train.py --raw img --option all --exp trial2 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --batch_size 8 --num_epoch 200 --use_ema --num_process_per_node 2 --save_content --input_path /mnt/aix21104/jw/data/syndiff/Train --output_path /mnt/aix21104/jw/ckpt --port_num 1234

python  CT_MAR_test.py --image_size 512 --exp trial1 --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --embedding_type positional  --z_emb_dim 256 --which_epoch 95 --input_path /mnt/aix21104/data/CT_MAR_RAW/Train --output_path /mnt/aix21104/jw/ckpt