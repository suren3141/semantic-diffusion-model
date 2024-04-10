for iter in 000600 001200 002000 003000; do
    for s_val in 1.2 1.5 2.0 2.5; do
        training_data=/mnt/dataset/dsb2018/
        testing_data=/mnt/dataset/dsb2018/
        model_path=/mnt/dataset/semantic-diffusion-model/dsb2018_32CH_128patch/ema_0.9999_${iter}.pt \
        out_dir=/mnt/dataset/dsb2018/out_sdm_32CH_128patch/output_s${s_val}_ema_0.9999_${iter}
        echo $out_dir
        python image_sample.py \
        --data_dir $testing_data \
        --dataset_mode dsb2018 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 128 \
        --learn_sigma True --noise_schedule linear --num_channels 32 --num_head_channels 32 --num_res_blocks 2 \
        --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 1 --class_cond False \
        --use_hv_map False --no_instance True --batch_size 2 --num_samples 10 --in_channels 1 \
        --model_path $model_path \
        --results_path $out_dir --s $s_val
    done
done


