import numpy as np

cm = np.array([1,2,3,4,5,6,7,7,7,7])

cm[cm != 7] = 0
cm[cm == 7] = 1

a = (["E:\\"],["F:\\"])
b = (["D:\\"],["home"])
A = zip(a,b)
print A