import numpy as np


def cvt2col(f):
    '''
    功能：从Kik-net原始地震动数据中提取加速度数据，进行格式转换，存储在'.acc'文件中。其中，不同分量文件名后部不同，文件名最后的'005'和'010'分别代表dt为0.005s和0.01s

    --Input
    -filename: string，Kik-net原始地震动文件名

    --Return
    -acc: numpy.array(n)，加速度数据
    -dt: float, 数据采样时间间隔
    '''
    for i in range(10):
        line = f.readline().decode()
    
    line = f.readline().decode()
    freq = line.split()[-1]
    freq = int(freq[:-2])
    dt = 1 / freq
    for i in range(2):
        line = f.readline().decode()

    line = f.readline().decode()
    line = line.split()[-1]
    line = line.split('/')
    scft = float(line[0][:-5]) / float(line[1])
    line = f.readline().decode()
    maxa = float(line.split()[-1])
    for i in range(2):
        line = f.readline().decode()

    acc = f.read()
    acc = acc.split()
    acc = [scft * float(a) for a in acc]
    acc = np.array(acc)
    acc = acc - np.mean(acc)
    f.close()
    return acc, dt