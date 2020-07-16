import numpy as np
import random
import time
from tqdm import trange

def buf_test_uint8 (test_n, batch_n, shape):
    buffer = np.random.randint(0,255,shape,dtype=np.uint8)
    for _ in trange(test_n):
        randidx = random.sample(range(shape[0]),batch_n)
        chosenones = buffer[randidx].astype(np.float32)/255

def buf_test_float32 (test_n, batch_n, shape):
    buffer = np.random.random_sample(shape).astype(np.float32)
    for _ in trange(test_n) :
        randidx = random.sample(range(shape[0]),batch_n)
        chosenones = buffer[randidx]

# DO NOT USE THIS : this was not the case
def buf_test_float32_cont (test_n, batch_n, shape):
    buffer = np.random.random_sample(shape).astype(np.float32)
    for _ in trange(test_n) :
        randidx = random.randrange(0,shape[0] - batch_n)
        chosenones = buffer[randidx:randidx+batch_n]

if __name__ == "__main__":
    st = time.time()
    buf_test_float32(100000, 32, (200000,2,2,100,3))
    print('all random: {}'.format(time.time()-st))
    st = time.time()
    buf_test_float32_cont(1000000, 32, (200000,2,2,100,3))
    print('random continuous: {}'.format(time.time()-st))