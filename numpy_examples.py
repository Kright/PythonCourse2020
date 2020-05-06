import numpy as np

print(np.array(1))
print(np.array([[1, 1]]))
arr = np.array([[1, 2, 3, 4]], dtype=np.uint8)

print(3 * np.ones(shape=(2, 3)))

a = np.ones(shape=(3, 1))
b = np.ones(shape=(1, 2))


arr = np.arange(6).reshape(2, 3)

print(np.stack([arr, arr]))
print(np.concatenate([arr, arr]))
print(np.concatenate([arr, arr], axis=1))

arr = np.arange(7)
arr2 = np.array(arr[::-1])

p = np.random.permutation(7)
print(p)

arr = arr[p]
arr2 = arr2[p]

print(arr)
print(arr2)

print(arr2[[0, 1, 2]])

arr = np.array([0, 1, 0, 0, 2, 0])

kernel = np.array([1, -1])

a = np.stack([arr[:-1], arr[1:]])
print(a)
print(a.shape)
print(kernel.shape)

a2 = a * kernel[:, np.newaxis]
print(a2)

result = np.sum(a2, axis=0)

print(result)
print(arr[: -1] - arr[1:])

arr[0, 1]
