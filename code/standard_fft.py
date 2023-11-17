# 直接进行DFT计算
import numpy as np

def standard_fft(x, N):
    """直接根据矩阵计算DFT

    args:
        x: 需要计算DFT的复序列，一维数组
        N: 需要计算的DFT点数的阶数，即点数为2^N
    outputs:
        dft_results: 结果序列

    """

    if N >= 15:
        return standard_fft_(x, N)

    x_len = x.size
    dft_len = 2**N
    if dft_len < x_len:
        real_N = np.ceil(np.log2(x_len))
        print(f"输入序列的长度{x_len}超过DFT的点数{dft_len}，将按照N={real_N}调用")
        return standard_fft(x, real_N)

    # 补0到相应长度
    real_x = np.concatenate([x.squeeze(), np.zeros(dft_len - x_len)], 0)


    # 定义W矩阵
    W_N = np.exp(-2j * np.pi / dft_len)
    matrix_W = W_N ** np.matmul(
        np.expand_dims(np.arange(dft_len), 1), np.expand_dims(np.arange(dft_len), 0)
    )

    dft_results = np.matmul(matrix_W, real_x)

    return dft_results

def standard_fft_(x, N):
    """ 减小内存占用的版本

    由于直接定义矩阵会爆内存，因此这里每次只计算矩阵的一行，减小内存占用
    
    """
    x_len = x.size
    dft_len = 2**N
    if dft_len < x_len:
        real_N = np.ceil(np.log2(x_len))
        print(f"输入序列的长度{x_len}超过DFT的点数{dft_len}，将按照N={real_N}调用")
        return standard_fft(x, real_N)

    # 补0到相应长度
    real_x = np.concatenate([x.squeeze(), np.zeros(dft_len - x_len)], 0)
    real_x = real_x.astype(np.complex_)
    dft_results = np.zeros(dft_len)*complex(1,0)
    # 定义W矩阵
    W_N = np.exp(-2j * np.pi / dft_len)
    for i in range(dft_len):
        vec_W_N = W_N ** (np.arange(dft_len)*i)
        dft_results[i] = np.matmul(vec_W_N, real_x)

    return dft_results


if __name__ == "__main__":
    x = np.random.rand(1, 10)
    N = 4
    print(standard_fft_(x, N))
    # 对比标准的fft：
    y = np.fft.fft(x, 2**4)
    print(y)
