# 使用当前时间作为文件名在 auto_run 文件夹下创建一个空的文本文件
filename="./auto_run_res.txt"
# 删除文件（如果存在）
if [ -f "$filename" ]; then
    rm "$filename"
    echo "文件 $filename 已删除"
else
    echo "文件 $filename 不存在，无需删除"
fi

touch "$filename"

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" >> "$filename"
echo "++++++++++++++++++++++++++++++++++++      shell out      ++++++++++++++++++++++++++++++++++++" >> "$filename"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" >> "$filename"


# only compute mIoU

python main.py --model vit_base_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_base \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_base/2025-01-13-19-57-06-ckpt/checkpoint_best.pth \
                --att_thr 0.05

python main.py --model vit_base_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_base \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_base/2025-01-13-19-57-06-ckpt/checkpoint_best.pth \
                --att_thr 0.1

python main.py --model vit_base_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_base \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_base/2025-01-13-19-57-06-ckpt/checkpoint_best.pth \
                --att_thr 0.2

python main.py --model vit_base_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_base \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_base/2025-01-13-19-57-06-ckpt/checkpoint_best.pth \
                --att_thr 0.3

python main.py --model vit_base_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_base \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_base/2025-01-13-19-57-06-ckpt/checkpoint_best.pth \
                --att_thr 0.4

python main.py --model vit_base_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_base \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_base/2025-01-13-19-57-06-ckpt/checkpoint_best.pth \
                --att_thr 0.5

python main.py --model vit_base_patch16_224_attmap \
                --batch-size 1 \
                --data-set CUB \
                --img-list cub \
                --data-path /data/c425/tjf/datasets/CUB_200_2011/ \
                --layer-index 12 \
                --output_dir results_vit_base \
                --gen_bounding_boxes \
                --ckpt /data/c425/tjf/attention_map_vit/results_vit_base/2025-01-13-19-57-06-ckpt/checkpoint_best.pth \
                --att_thr 0.6


# 读取文件内容并输出到终端
cat "$filename"
















