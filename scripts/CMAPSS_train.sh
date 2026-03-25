export CUDA_VISIBLE_DEVICES=0


for seed in 0 1 2 3 4
do

for data_id in FD001
do

for M_name in  DiDA_Net
do

for length in 50
do

for rate in 0.00001
do

for d_model in 64
do

for d_ff in 128
do


python -u main.py \
  --task 'normal'\
  --seed $seed\
  --dataset_name 'CMAPSS'\
  --Data_id_CMAPSS $data_id\
  --train True\
  --resume False\
  --input_length $length\
  --batch_size 64\
  --d_model $d_model\
  --d_ff $d_ff\
  --dropout 0.1\
  --model_name $M_name\
  --train_epochs 150\
  --learning_rate $rate\

done

done

done

done

done

done

done



