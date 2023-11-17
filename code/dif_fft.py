# 基2 DIF-FFT算法
# 先做蝶形运算，然后二进制倒排
import numpy as np
from standard_fft import standard_fft
from dit_fft import dit_fft


def dif_fft(x, N):
    """基2 DIF-FFT算法

    args:
        x: 输入的复序列
        N: 要做FFT的点数的对数（以2为底），即为fft的级数
        （要求 2^N >= length(x) ）
    outputs:
        input_x: 输出的dft结果
    """
    x_len = x.size
    fft_len = 2**N
    if fft_len < x_len:
        real_N = np.ceil(np.log2(x_len))
        print(f"要做fft序列长度{x_len}超过点数{fft_len}，将按照N={real_N}进行fft")
        return dif_fft(x, real_N)

    # 补0到相应长度
    real_x = np.concatenate([x.squeeze(), np.zeros(fft_len - x_len)], 0)
    real_x = real_x.astype(np.complex_)
    # pdb.set_trace()
    for i in range(N):
        vec_W_N = np.exp(-2j * np.pi / (2 ** (N - i)) * np.arange(2 ** (N - 1 - i)))
        for j in range(0, fft_len, 2 ** (N - i)):
            for k in range(2 ** (N - 1 - i)):
                # 遍历一个里面的 2^i 个蝶形
                real_x[j + k], real_x[j + k + 2 ** (N - 1 - i)] = (
                    real_x[j + k] + real_x[j + k + 2 ** (N - 1 - i)] * complex(1, 0),
                    (real_x[j + k] - real_x[j + k + 2 ** (N - 1 - i)]) * vec_W_N[k],
                )

    # 二进制逆序
    # 将输入序列按照二进制倒排
    input_x = [
        real_x[i]
        for i in list(
            map(lambda x: int("{:0{}b}".format(x, N)[::-1], 2), np.arange(fft_len))
        )
    ]

    return input_x


if __name__ == "__main__":
    x = np.random.rand(1, 6)
    N = 3
    # 对比标准的fft：
    y = np.fft.fft(x, 2**3)
    print(y)
    y2 = dif_fft(x, N)
    print(y2)
