python train_mnist.py --device=cuda:0 --log-interval=-1 --save-model &
python train_mnist.py --device=cuda:1 --log-interval=-1 --save-model &
python train_mnist.py --device=cuda:2 --log-interval=-1 --save-model &
python train_mnist.py --device=cuda:3 --log-interval=-1 --save-model &
wait
git checkout 7720b
python train_mnist.py --device=cuda:0 --log-interval=-1 --save-model &
python train_mnist.py --device=cuda:1 --log-interval=-1 --save-model &
python train_mnist.py --device=cuda:2 --log-interval=-1 --save-model &
python train_mnist.py --device=cuda:3 --log-interval=-1 --save-model &
wait

