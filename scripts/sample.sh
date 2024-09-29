# for iter in 030000 090000 150000 210000; do
for iter in 030000 040000 050000; do
    for s_val in 1.3 1.4 1.5 1.6; do
        # for cluster in 0 1 2 3 4 5 6 7 8 9; do
        # cluster=4
        num_channels=128
        num_head_channels=32
        steps=1000
        lr=1e-4
        scheduler="cosine"

        version="v1.4"
        mod="model"
        # mod="ema_0.9999_"

        testing_data=/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/__ResNet50_umap_n_components_3_random_state_42_hdbscan_min_samples_10_min_cluster_size_50_v1.2/6/

        model_name=patches_valid_${num_channels}.${num_head_channels}CH_${steps}st_${lr}lr_8bs_hvb_col_cos_clus6
        model_path=/mnt/dataset/semantic-diffusion-model/monuseg_128x128/${model_name}/${mod}${iter}.pt
        out_dir=/mnt/dataset/MoNuSeg/out_sdm_128x128/${model_name}/${version}_output_model_s${s_val}_${iter}
        echo $out_dir
        python image_sample.py \
        --data_dir $testing_data --dataset_mode monuseg --use_train True \
        --shuffle_masks False --match_struct True --match_app False \
        --attention_resolutions 32,16,8 --diffusion_steps ${steps} --image_size 128 \
        --learn_sigma True --noise_schedule ${scheduler} --num_channels $num_channels --num_head_channels $num_head_channels --num_res_blocks 2 \
        --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
        --num_classes 6 --class_cond False --use_hv_map True --use_col_map True --no_instance False \
        --batch_size 16 --num_samples 1000 \
        --model_path $model_path \
        --results_path $out_dir --s $s_val
        # python /mnt/dataset/MoNuSeg/src/combine_outputs.py $out_dir
    done
done


