export CUDA_VISIBLE_DEVICES=0

model_name=ADKLTS
model_id=ADKF

seq_len=96
learning_rate=0.001
batch_size=4
train_epochs=50
patience=3

python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/weather \
  --data_path weather.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 96 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --channel 21 \
  --var_num 21 \
 
python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/weather \
  --data_path weather.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 192 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --channel 21 \
  --var_num 21 \

python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/weather \
  --data_path weather.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 336 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --channel 21 \
  --var_num 21 \
 
python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/weather \
  --data_path weather.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 720 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --channel 21 \
  --var_num 21 \

python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/electricity \
  --data_path electricity.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 96 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --channel 321 \
  --var_num 321 \

python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/electricity \
  --data_path electricity.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 192 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --channel 321 \
  --var_num 321 \

python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/electricity \
  --data_path electricity.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 336 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 2 \
  --channel 321 \
  --var_num 321 \
 
python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/electricity \
  --data_path electricity.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 720 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience 1 \
  --batch_size 1 \
  --channel 321 \
  --var_num 321 \
 
 python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/traffic \
  --data_path traffic.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 96 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 1 \
  --channel 862 \
  --var_num 862 \
  
python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/traffic \
  --data_path traffic.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 192 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 1 \
  --channel 862 \
  --var_num 862 \

python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/traffic \
  --data_path traffic.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 336 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 1 \
  --channel 862 \
  --var_num 862 \

python -u /home/zyg/Project/ADKF-main/run.py \
  --is_training 1 \
  --model_id $model_id \
  --data custom \
  --root_path  /home/zyg/data/time_series/traffic \
  --data_path traffic.csv \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 720 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 1 \
  --channel 862 \
  --var_num 862 \
  
