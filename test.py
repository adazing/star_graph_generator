import numpy as np

print(np.ones((4,), dtype=np.uint16))

print(np.asarray([1,2,3,4,5]).size)
print(np.asarray([1,2,3,4,5])[2:4])

for x in range(5):
    for y in range(5,10):
        print(x, y)
        if y == 8:
            break