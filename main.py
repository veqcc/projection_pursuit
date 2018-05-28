import numpy as np

# 標本数
N = 1000

# Gauss分布を生成
x_gauss = np.random.randn(N)

# 一様分布を生成
x_uniform = np.random.rand(N) * 5.0 - 2.5

x = np.array([x_gauss, x_uniform])


def nomalize(b):
    return b / np.sqrt(b[0]**2 + b[1]**2)


def main():
    # g(s) = s^3 のとき
    # 初期のbを仮定
    b = nomalize(np.array([1, 0]))
    threshold = 0.0000000001
    diff = 1

    while diff > threshold:
        print(b)
        tmp_1, tmp_2 = 0, 0
        for i in range(N):
            tmp_1 += 3 * (np.dot(b, x[:, i]) ** 2)
            tmp_2 += x[:, i] * (np.dot(b, x[:, i]) ** 3)

        new_b = tmp_2 / N - tmp_1 * b / N
        diff = (b[0] - nomalize(new_b)[0])**2 + (b[1] - nomalize(new_b)[1])**2
        b = nomalize(new_b)


if __name__ == '__main__':
    main()
