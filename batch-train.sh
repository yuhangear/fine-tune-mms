#!/bin/bash


num_gpus=1

export CUDA_VISIBLE_DEVICES="0,1,2"
cmd="./slurm.pl --quiet --nodelist=node06"
$cmd --num-threads 6 --gpu 3 train.1.log    ./train.sh



# cmd="./utils/run.pl "
# $cmd  JOB=1:$num_gpus train_log/train.JOB.log    ./train.sh --rank JOB --num_gpus $num_gpus --num_workers $num_workers


#  source /home/yuhang001/.bashrc
#  conda activate wenet
# cd /home/yuhang001/w2023/wenet-eng-id/wenet
#  bash batch-train.sh > log1 2 1>&1 &