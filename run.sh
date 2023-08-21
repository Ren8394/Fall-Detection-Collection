export CUDA_VISIBLE_DEVICES='0'

python main.py --dataset FallAllD --model CNN --epochs 100 --location Wrist
python main.py --dataset UMAFall --model CNN --epochs 100 --location Wrist
python main.py --dataset UMAFall --model CNN --epochs 100 --location RightPocket

python main.py --dataset FallAllD --model LSTM --epochs 200 --location Wrist
python main.py --dataset UMAFall --model LSTM --epochs 200 --location Wrist
python main.py --dataset UMAFall --model LSTM --epochs 200 --location RightPocket

python main.py --dataset FallAllD --model CNNLSTM --epochs 150 --location Wrist
python main.py --dataset UMAFall --model CNNLSTM --epochs 150 --location Wrist
python main.py --dataset UMAFall --model CNNLSTM --epochs 150 --location RightPocket
