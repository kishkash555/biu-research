conda activate shahar
git checkout parametric-pytorch-hashnet1
python HashNet.py --device=cuda:0 --log-interval=-1 --seed=64 --activation=relu --save-model &
python HashNet.py --device=cuda:1 --log-interval=-1 --seed=32 --activation=relu --save-model &
python HashNet.py --device=cuda:2 --log-interval=-1 --seed=16 --activation=relu --save-model &
python HashNet.py --device=cuda:3 --log-interval=-1 --seed=8 --activation=relu --save-model &
wait
python HashNet.py --device=cuda:0 --log-interval=-1 --seed=64 --activation=relu --save-model &
python HashNet.py --device=cuda:1 --log-interval=-1 --seed=32 --activation=relu --save-model &
python HashNet.py --device=cuda:2 --log-interval=-1 --seed=16 --activation=relu --save-model &
python HashNet.py --device=cuda:3 --log-interval=-1 --seed=8 --activation=relu --save-model &

