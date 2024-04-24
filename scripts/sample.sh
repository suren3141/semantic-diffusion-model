# for iter in 030000 090000 150000 210000; do
for iter in 030000 020000 010000; do
    for s_val in 1.2 1.5 2.0 2.5; do
        for cluster in 0 1 2 3 4 5 6 7 8 9; do
        num_channels=128
        num_head_channels=64
        steps=200
        lr=1e-4
        # training_data=names_37_14/TCGA-18-5592-01Z-00-DX1
        testing_data=/mnt/dataset/MoNuSeg/patches_256x256_128x128/ResNet18_kmeans_10_v1.1/${cluster}/
        model_name=monuseg_patches_${num_channels}.${num_head_channels}CH_${steps}st_${lr}lr_8bs_hv_ResNet18_kmeans_10_v1.1_4
        model_path=/mnt/dataset/semantic-diffusion-model/monuseg/${model_name}/model${iter}.pt \
        out_dir=/mnt/dataset/MoNuSeg/out_sdm/${model_name}/ResNet18_kmeans_10_v1.1/${cluster}/output_s${s_val}_${iter}
        echo $out_dir
        python image_sample.py \
        --data_dir $testing_data \
        --dataset_mode monuseg --attention_resolutions 32,16,8 --diffusion_steps ${steps} --image_size 256 \
        --learn_sigma True --noise_schedule linear --num_channels $num_channels --num_head_channels $num_head_channels --num_res_blocks 2 \
        --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 2 --class_cond False \
        --use_hv_map True --no_instance True --batch_size 16 --num_samples 100 \
        --model_path $model_path \
        --results_path $out_dir --s $s_val
        python /mnt/dataset/MoNuSeg/combine_outputs.py $out_dir
        done
    done
done


