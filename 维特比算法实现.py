import numpy as np
T = 8
N = 3
# 状态转移矩阵A
A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]
# 观测概率矩阵B
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
# 初始状态概率向量pi
pi = [0.2, 0.3, 0.5]
# 初始观测序列O
O = ['红', '白', '红', '红', '白', '红', '白', '白']
# 转换O为0,1标志（数组o中，1代表白色，0代表红色）
o = np.zeros(T, int) # 初始化为int型0数组
for i in range(T):
    if O[i] == '白':
        o[i] = 1
    else:
        o[i] = 0

# 动态规划求概率最大路径（最优路径）
# i=1,2,...,N
delta = np.zeros((T, N))
psi = np.zeros((T, N), int)
a = []
i_star = np.zeros(T, int)
# 初始化求  delta(1,i),psi(1,i)
# delta(1,i)=pi(i)*b(i,o(i))
# psi(1,i)=0,i=1,2,...,N
for i in range(N):
    delta[0][i] = pi[i] * B[i][o[0]]
    psi[0][i] = 0

# 递推,对t=2,3,4....(T)
# delta(t,i)=max{1<=j<=N} [delta(t-1,j)a(j,i))]b(i,o(t))
# psi(t,i)=argmax{1<=j<=N} [delta(t-1,j)a(j,i)]
for t in range(T-1):
    t = t + 1
    for i in range(N):
        for j in range(N):
            a.append(delta[t-1][j] * A[j][i])
        delta[t][i] = np.max(a) * B[i][o[t]]
        psi[t][i] = np.argmax(a, axis=0)
        a = []
psi = psi + 1

# 终止
# P_star=max{1<=j<=N} delta(T,i)
# i_star_Transpose=argmax{1<=j<=N} [delta(T,i)]
P_star = np.max(delta[T-1])
i_star[T-1] = np.argmax(delta[T-1], axis=0) +1

# 求最优路径回溯，对t=T-1,T-2,...,1
# i_star_t=psi(t+1,i_star(t+1))
for t in range(T-1):
    t = T - t - 2
    a = t + 1
    b = i_star[t+1]-1
    i_star[t] = psi[a][b]

print('最优路径： ')
print(i_star)
print('最优路径的概率：')
print(P_star)
