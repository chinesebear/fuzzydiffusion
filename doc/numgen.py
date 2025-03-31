import random
def numgen(a,b,n):
    str = ""
    for _ in range(n):
        f = random.uniform(a, b)
        f = round(f,2)
        str = str + f"{f},"
    print(str)
# numgen(6.21,7.20, 10)
numgen(0.01,0.19, 10)
# numgen(29.30,29.61, 10)
# numgen(6.21,7.20, 20)
# numgen(0.42,0.48, 20)
# numgen(29.30,29.61, 20)