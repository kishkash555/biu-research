import dynet as dy
import numpy as np

if __name__ == "__main__":
    pc = dy.ParameterCollection()
    a = pc.add_parameters((1,1))
    b = pc.add_parameters((1,1))

    # now generate data
    ta = 10
    tb = 4
    k = 100
    x = np.random.uniform(0,5,size=k)
    y = ta * x + 0.755 * np.random.randn(k) + tb

    print('initial a,b: {}, {}'.format(a.scalar_value(), b.scalar_value()))
    trainer = dy.SimpleSGDTrainer(pc)
    for ep in range(3):
        for t in range(k):
            dy.renew_cg()
            xi = dy.inputVector([x[t]])
            yi = dy.inputVector([y[t]])
            a2 = dy.inputVector([a.scalar_value()])
            b2 = dy.inputVector([b.scalar_value()])
            loss = dy.square(xi * a2 + b2 - yi)
            loss.backward()
            trainer.update()

    print('final a,b: {}, {}'.format(a.scalar_value(), b.scalar_value()))

#    print(x.shape)
#    print(y.shape)
#    print(x[:5])
#    print(y[:5])