python main.py --model vit_small_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_small \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_small/2025-01-11-21-00-26-ok-epoch60/checkpoint_best.pth