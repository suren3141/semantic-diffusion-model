# These are the parameters used for synthesis
iter=040000
s_val=1.5
# Use the same parameter as training
num_channels=128
num_head_channels=32
steps=1000
lr=1e-4
scheduler="cosine"
image_size=128

version="v1.4"
mod="model"
# mod="ema_0.9999_"

# Edit path to dataset path (with MoNuSegTrainingData and MoNuSegTestData folders)
testing_data=/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/__ResNet50_umap_n_components_3_random_state_42_hdbscan_min_samples_10_min_cluster_size_50_v1.2/6/
# Model name used during training.
model_name=patches_valid_${num_channels}.${num_head_channels}CH_${steps}st_${lr}lr_8bs_hvb_col_cos_clus6
model_path=/mnt/dataset/semantic-diffusion-model/monuseg_128x128/${model_name}/${mod}${iter}.pt
# Path to save output images.
out_dir=/mnt/dataset/MoNuSeg/out_sdm_128x128/${model_name}/${version}_output_model_s${s_val}_${iter}
echo $out_dir
# New parameters were introduced for synthesis, and can be adjusted as needed
# --use_train True : If set to True, MoNuSegTrainingData images are used for image synthesis. If False, MoNuSegTestData
# --shuffle_masks False : Set to True to randomly shuffle structure for each appearance. 
# --match_struct True : Set to True to find structure based on structure matching. 
# --match_app False : Set to True to find appearance based on appearance matching
python image_sample.py \
--data_dir $testing_data --dataset_mode monuseg --use_train True \
--shuffle_masks False --match_struct True --match_app False \
--attention_resolutions 32,16,8 --diffusion_steps ${steps} --image_size $image_size \
--learn_sigma True --noise_schedule ${scheduler} --num_channels $num_channels --num_head_channels $num_head_channels --num_res_blocks 2 \
--resblock_updown True --use_fp16 True --use_scale_shift_norm True \
--num_classes 6 --class_cond False --use_hv_map True --use_col_map True --no_instance False \
--batch_size 16 --num_samples 1000 \
--model_path $model_path \
--results_path $out_dir --s $s_val
# python /mnt/dataset/MoNuSeg/src/combine_outputs.py $out_dir


