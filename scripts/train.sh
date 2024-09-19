# for iter in 030000 090000 150000 210000; do
num_channels=128
num_head_channels=64
steps=1000
lr=1e-4
batch_size=8
# training_data=names_37_14/TCGA-18-5592-01Z-00-DX1
# training_data=/mnt/dataset/MoNuSeg/patches_256x256_128x128/ResNet18_kmeans_10_v1.1/4/
training_data=/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/__ResNet50_umap_n_components_3_random_state_42_hdbscan_min_samples_10_min_cluster_size_50_v1.2/6/10ss/MoNuSegTrainingData/
model_name=patches_valid_${num_channels}.${num_head_channels}CH_${steps}st_${lr}lr_8bs_hvb_col_cos_clus6_10ss
model_path=/mnt/dataset/semantic-diffusion-model/monuseg_128x128/${model_name} \
# out_dir=/mnt/dataset/MoNuSeg/out_sdm/${model_name}/ResNet18_kmeans_10_v1.1/4/output_s${s_val}_${iter}
echo $model_path
OPENAI_LOGDIR="$model_path" \
OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard" \
python image_train.py 
--data_dir $training_data \
--dataset_mode monuseg \
--image_size 128 \
--lr $lr --batch_size $batch_size \
--attention_resolutions 32,16,8 \
--diffusion_steps $steps \
--learn_sigma True --noise_schedule cosine --num_channels $num_channels --num_head_channels $num_head_channels --num_res_blocks 2 \
--resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True \
--num_classes 6 --class_cond False --use_hv_map True --use_col_map True --no_instance False \
--save_interval 005000 --drop_rate 0.2

