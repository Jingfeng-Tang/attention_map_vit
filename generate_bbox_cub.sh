# only compute mIoU

python main.py --model vit_small_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_small \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_small/2025-01-11-21-00-26-ok-epoch60/checkpoint_best.pth \
                --att_thr 0.05

python main.py --model vit_tiny_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_tiny \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_tiny/2025-01-13-19-14-34-ckpt/checkpoint_best.pth \
                --att_thr 0.2

python main.py --model vit_base_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_base \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_base/2025-01-13-19-57-06/checkpoint_best.pth \
                --att_thr 0.05

# compute mIoU and generate attention maps

python main.py --model vit_small_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_small \
                --gen_bounding_boxes \
                --gen_attention_maps \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_small/2025-01-11-21-00-26-ok-epoch60/checkpoint_best.pth




# compute mIoU and generate attention maps_boxes

python main.py --model vit_small_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_small \
                --gen_bounding_boxes \
                --gen_maps_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_small/2025-01-11-21-00-26-ok-epoch60/checkpoint_best.pth