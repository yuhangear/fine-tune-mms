#!/usr/bin/env bash

cmd="./slurm.pl --quiet --nodelist=node01"
#/home/yuhang001/w2023/wenet-eng-id/wenet/data/ntu-conversation/data.list /home/yuhang001/w2023/wenet-eng-id/wenet/data/yt_test_v2/data.list /home/yuhang001/w2023/wenet-eng-id/wenet/data/King_test/data.list
yt_test=" /home/yuhang001/w2023/wenet-eng-id/wenet/data/yt_test/data.list "
yt_test_v2="/home/yuhang001/w2023/wenet-eng-id/wenet/data/yt_test_v2/data.list"
yt_test3="/home/yuhang001/w2023/wenet-eng-id/wenet/data/yt_3_test/data.list"
king_test="/home/yuhang001/w2023/wenet-eng-id/wenet/data/King_test/data.list"
speac_co_1_test="/home/yuhang001/w2023/wenet-eng-id/wenet/data/speac.co_1_test/data.list"
speaco_co_2_test="/home/yuhang001/w2023/wenet-eng-id/wenet/data/speaco.co_2_test/data.list"
id_read_test="/home/yuhang001/w2023/wenet-eng-id/wenet/data/id_read_test/data.list"
ntu_cov="/home/yuhang001/w2023/wenet-eng-id/wenet/data/ntu-conversation/data.list"
test_clean="/home/yuhang001/yuhang001/espnet/egs2/librispeech_100/asr1/data/test_clean/data.list"

# test_dir=" /home/yuhang001/w2023/wenet-eng-id/wenet/data/yt_test/data.list " # /home/yuhang001/w2023/wenet-eng-id/wenet/data/yt_3_test/data.list /home/yuhang001/w2023/wenet-eng-id/wenet/data/speac.co_1_test/data.list /home/yuhang001/w2023/wenet-eng-id/wenet/data/speaco.co_2_test/data.list "
# ${speac.co_1_test} ${speaco.co_2_test}  $yt_test_v2  $yt_test3 $king_test  $id_read_test  $ntu_cov

#$yt_test_v2  $yt_test3 $king_test $speac_co_1_test $speaco_co_2_test  $id_read_test $ntu_cov  

#$yt_test_v2  $yt_test3 $king_test $speac_co_1_test $speaco_co_2_test  $id_read_test $ntu_cov
:>all_wer
for i in $ntu_cov    ; do  

# $cmd --num-threads 5 --gpu 1 ./decode_log_gpu_cpu5_man python decode.py $i
python decode.py $i 
wav_name=$( basename  $(dirname "$i"))
echo $wav_name >> all_wer
python tools/compute-wer.py --char 1 --v 1  "./decoder_${wav_name}_ref" "./decoder_${wav_name}_reg" > ${wav_name}_wer
tail  ${wav_name}_wer >> all_wer
wait 
done 