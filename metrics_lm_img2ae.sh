#!/bin/bash

gpu=0
exp=expts/6/lm_img2ae_lamp_emd \
eval_set=valid
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
declare -a categs=("lamp")

for cat in "${categs[@]}"; do
	echo python metrics_lm_img2ae.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --mode lm --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 128
	python metrics_lm_img2ae.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --mode lm --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 128
done

declare -a categs=("lamp")
for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics/${eval_set}/${cat}.csv
	echo
done
