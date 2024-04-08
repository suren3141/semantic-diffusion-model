for iter in 030000 090000 150000 210000; do
    for s_val in 1.2 1.5 2.0 2.5; do
        training_data=names_37_14/TCGA-18-5592-01Z-00-DX1
        testing_data=TCGA-AC-A2FO-01A-01-TS1
        model_path=/mnt/dataset/semantic-diffusion-model/monuseg_patches_128CH_hv/names_37_14/TCGA-18-5592-01Z-00-DX1/ema_0.9999_${iter}.pt \
        out_dir=/mnt/dataset/MoNuSeg/out_sdm_128CH_hv/patches_256x256_128x128/names_37_14/TCGA-18-5592-01Z-00-DX1/TCGA-AC-A2FO-01A-01-TS1/output_s${s_val}_ema_0.9999_${iter}
        echo $out_dir
        python image_sample.py \
        --data_dir /mnt/dataset/MoNuSeg/patches_256x256_128x128/names_37_14/TCGA-AC-A2FO-01A-01-TS1/ \
        --dataset_mode monuseg --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 \
        --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 32 --num_res_blocks 2 \
        --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 2 --class_cond False \
        --use_hv_map True --no_instance True --batch_size 2 --num_samples 10 \
        --model_path 
        --results_path $out_dir --s $s_val
    done 
done

