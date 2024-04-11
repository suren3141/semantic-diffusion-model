OPENAI_LOGDIR="/mnt/dataset/semantic-diffusion-model/monuseg_patches_128.64CH_200st_hv/" \
OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard" \
    python image_train.py \
    --data_dir /mnt/dataset/MoNuSeg/patches_256x256_128x128/ \
    --dataset_mode monuseg --lr 1e-4 --batch_size 2 --attention_resolutions 32,16,8 \
    --diffusion_steps 200 --image_size 256 --learn_sigma True --noise_schedule linear \
    --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True \
    --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 2 --class_cond False \
    --use_hv_map True --no_instance True --save_interval 010000 --drop_rate .2 