conda activate shahar
git checkout parametric-pytorch-hashnet1
mkdir -p results
python HashNet.py --device=cuda:0 --log-interval=-1 --seed=64 --activation=relu --save-model --batch-size=16 --test-batch-size=16 &
python HashNet.py --device=cuda:1 --log-interval=-1 --seed=32 --activation=relu --save-model --batch-size=16 --test-batch-size=16 &
python HashNet.py --device=cuda:2 --log-interval=-1 --seed=16 --activation=relu --save-model --batch-size=32 --test-batch-size=32 &
python HashNet.py --device=cuda:3 --log-interval=-1 --seed=8 --activation=relu --save-model --batch-size=32 --test-batch-size=32 &
wait
python HashNet.py --device=cuda:0 --log-interval=-1 --seed=64 --activation=relu --save-model --batch-size=16 --test-batch-size=16 &
python HashNet.py --device=cuda:1 --log-interval=-1 --seed=32 --activation=relu --save-model --batch-size=16 --test-batch-size=16 &
python HashNet.py --device=cuda:2 --log-interval=-1 --seed=16 --activation=relu --save-model --batch-size=32 --test-batch-size=32 &
python HashNet.py --device=cuda:3 --log-interval=-1 --seed=8 --activation=relu --save-model --batch-size=32 --test-batch-size=32 &

