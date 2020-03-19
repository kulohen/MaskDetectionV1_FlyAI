# 首先获得Iterator对象:
it = iter([1, 2, 3, 4, 5])
# 循环:
a=0
while True:
    a += 1
    try:
        # 获得下一个值:

        x = next(it)
        print('a', a)
        print(x)
        if a == 15 :

            break
    except StopIteration:
        print('error')
        it = iter([1, 2, 3, 4, 5])
        x = next(it)
        print('a', a)
        print(x)