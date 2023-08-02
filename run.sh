export CUDA_VISIBLE_DEVICES='0'

python main.py --dataset FallAllD --model CNN --epochs 100

python main.py --dataset SisFall --model CNN --epochs 100
 
python main.py --dataset UMAFall --model CNN --epochs 100

python main.py --dataset FallAllD --model LSTM --epochs 250

python main.py --dataset SisFall --model LSTM --epochs 250
 
python main.py --dataset UMAFall --model LSTM --epochs 250

python main.py --dataset FallAllD --model CNNLSTM --epochs 250

python main.py --dataset SisFall --model CNNLSTM --epochs 250
 
python main.py --dataset UMAFall --model CNNLSTM --epochs 250
