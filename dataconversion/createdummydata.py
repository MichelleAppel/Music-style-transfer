import os
import numpy as np

width = 128
height = 128

domains = 3
number = 10

dirname = './output/dummydata'

if not os.path.exists(dirname):
    os.mkdir(dirname)

for d in range(domains):
    if not os.path.exists(dirname+'/'+str(d)):
        os.mkdir(dirname+'/'+str(d))
    for n in range(number):
        spectrogram = np.random.rand(height, width)*10
        np.save(dirname+'/'+str(d)+'/'+str(n), spectrogram)