import math
def mincostfn(N,A,B,C,D):
    mincost = float('inf')
    for i in range(0,math.ceil(N/A)+1):
        fromA = i*A
        remaining = max(0,N-fromA)

        j = math.ceil(remaining/C)

        total_cost = i*B + j*D

        mincost = min(total_cost,mincost)

    return mincost
N, A, B, C, D = 17, 5, 10, 8, 14
print(mincostfn(N,A,B,C,D))