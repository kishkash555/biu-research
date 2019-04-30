conda activate shahar
git checkout parametric-pytorch-hashnet1
python HashNet.py --device=cpu --log-interval=-1 --seed=4 --activation=relu --save-model --batch-size=32 --test-batch-size=32 &
python HashNet.py --device=cpu --log-interval=-1 --seed=2 --activation=relu --save-model --batch-size=32 --test-batch-size=32 &
python HashNet.py --device=cpu --log-interval=-1 --seed=1 --activation=relu --save-model --batch-size=32 --test-batch-size=32 &
python HashNet.py --device=cpu --log-interval=-1 --seed=4 --activation=relu --save-model --batch-size=32 --test-batch-size=32 &
python HashNet.py --device=cpu --log-interval=-1 --seed=2 --activation=relu --save-model --batch-size=32 --test-batch-size=32 &
python HashNet.py --device=cpu --log-interval=-1 --seed=1 --activation=relu --save-model --batch-size=32 --test-batch-size=32 &


