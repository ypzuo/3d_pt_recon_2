#!/bin/bash

gpu=0
exp=expts/6/lm_img2ae_table_emd \
eval_set=test
dataset=pix3d
data_dir_imgs=/home/ubuntu/3Dreconstruction/Dataset/pix3d
data_dir_pcl=/home/ubuntu/3Dreconstruction/Dataset/pix3d/pointclouds
declare -a categs=("table")

for cat in "${categs[@]}"; do
	python pix_metrics.py \
		--gpu $gpu \
		--dataset $dataset \
		--data_dir_imgs ${data_dir_imgs} \
		--data_dir_pcl ${data_dir_pcl} \
		--exp $exp \
		--category $cat \
		--load_best \
		--bottleneck 512 \
		--bn_decoder \
		--eval_set ${eval_set} \
		--batch_size 1 \
		--visualize 
done

clear
declare -a categs=("table")
for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics_$dataset/${eval_set}/${cat}.csv
	echo
done