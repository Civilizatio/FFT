# 基2 DIT-FFT算法
# 先将输入序列变为二进制倒序，然后根据每一级进行
# 的 2^(N-1) 个蝶形运算（N为序列长度的对数，也为最终的级数）
import numpy as np

from standard_fft import standard_fft


# 对输入序列进行二进制倒排
def reverse_bit(x, N):
    """对输入的索引进行二进制倒排，输出倒排后的索引

    args:
        x: 输入索引，0~2^N
        N: 二进制位数（高位不够则补零）
    """

    bits = "{:0{}b}".format(x, N)  # N位二进制数
    return int(bits[::-1], 2)  # 逆序后转化为10进制数


def dit_fft(x, N):
    """基2 DIT-FFT算法

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
        return dit_fft(x, real_N)

    # 补0到相应长度
    real_x = np.concatenate([x.squeeze(), np.zeros(fft_len - x_len)], 0)

    # 将输入序列按照二进制倒排
    input_x = [
        real_x[i]
        for i in list(
            map(lambda x: int("{:0{}b}".format(x, N)[::-1], 2), np.arange(fft_len))
        )
    ]

    # 每一级进行蝶形运算
    for i in range(N):
        vec_W_N = np.exp(-2j * np.pi / (2 ** (i + 1)) * np.arange(2**i))
        for j in range(0, fft_len, 2 ** (i + 1)):
            # 遍历 2^(N-i) 个蝶形
            for k in range(2**i):
                # 遍历一个里面的 2^i 个蝶形
                input_x[j + k], input_x[j + k + 2**i] = (
                    input_x[j + k] + vec_W_N[k] * input_x[j + k + 2**i],
                    input_x[j + k] - vec_W_N[k] * input_x[j + k + 2**i],
                )

    # 返回
    return input_x


if __name__ == "__main__":
    x = np.random.rand(1, 10)
    N = 4
    print(standard_fft(x, N))
    # 对比标准的fft：
    y = np.fft.fft(x, 2**4)
    print(y)
    y2 = dit_fft(x, N)
    print(y2)
