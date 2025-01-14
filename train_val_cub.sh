python main.py --model vit_small_patch16_224_attmap \
                --batch-size 128 \
                --epochs 60 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_small \
                --finetune /data/c425/tjf/attention_map_vit/pretrained/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz

python main.py --model vit_tiny_patch16_224_attmap \
                --batch-size 128 \
                --epochs 60 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_tiny \
                --finetune /data/c425/tjf/attention_map_vit/pretrained/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz

python main.py --model vit_base_patch16_224_attmap \
                --batch-size 128 \
                --epochs 60 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_base \
                --finetune /data/c425/tjf/attention_map_vit/pretrained/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz