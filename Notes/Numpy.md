# Numpy
 ## For ML and Data Science

1. We use **Numpy** to perform mathematical operations on arrays and matrices if the arrays are 2d 
2. To start using `numpy` install it in your system with -:
```python
pip install numpy
```

3. To add the library to your code assign the library an alias like `np`
```python 
import numpy as np
```

4. To define an array in `numpy` we store the value of array in a variable and declare the array with the help of `np.array`
```python 
import numpy as np

myarray = np.array([2,5,3,7], np.int8 ) # np.int8 defines the data type

# 8 in int8 means the integer will be of 8 bit 
```

5. To access an already defined array we can 
```python 
myarr = np.array([[3,5,2,4],[7,4,9,2]])

print(myarr[0,1]) # accesses the array in 0th row and 1st column
```

6. To find the shape or the rows and columns in an array we use the `shape` function
```python 
arr1 = np.array([[3,4,9,1]])

print(arr1.shape) # will print out the (rows,column) of the array
```

7. To find the data type of the array 
```python 
arr1.dtype # int8 in this case 
```

8. There are 5 ways to create arrays using **numpy**
    * Conversion from other python **structures**(list,tuples)
    * **Intrinsic** **numpy** array creation objects (eg. arange, ones, zeros etc)
    * Reading arrays from the **disk**, either form **standard or custom formats**
    * Creating arrays from **raw bytes** through the use of **strings** or **buffers** 

9. Here are various methods to create arrays with examples 

- Conversion from `list/tuples`
```python
listarray = np.array([[2,4,5], [8,3,1], [9,4,5]])

#this is using list or tuples 
```

- Using `zeros` to create arrays
```python 
zeroarray = np.zeros((2,5)) # will create an array of 2 rows and 5 columns with each entry being zero 
```

- Using `arange` to create arrays
```python
rangearr = np.arange(15)
#creates a numpy array with 15 characters from 0 to 14
```

- Using linspace to create arrays
```python
lspace = np.linspace(1,6,12)
# this will create an array of 12 characters and with each element being in between 1 and 6 
#equally linearly spaced
```

- Using `identity matrix`
```python
ide = np.identity(20)

#this will give out an identity matrix of 20 x 20 
```

- Using `arange` and then reshape
```python 
arr = np.arange(99)
# this prints out a matrix with 99 elements with gap 1

arr.reshape(3,33) # this reshapes the matrix with 3 rows and 33 columns
```

- Using `ravel` to convert back the reshaped matrix into 1D matrix
```python
arr.ravel()
#returns back to 1D matrix with 1 row and 99 columns
```


10. Axis is basically used to tell about rows and columns they start from 0 and go on and **axis 0 is for rows** and **axis 1** **is for** **columns** 

![axis in numpy](https://cdn-coiao.nitrocdn.com/CYHudqJZsSxQpAPzLkHFOkuzFKDpEHGF/assets/static/optimized/rev-85bf93c/wp-content/uploads/2018/12/numpy-arrays-have-axes_updated_v2-1024x525.png)

11. **1D** array has **Axis 0** while **2D array has Axis 0 and Axis 1** 
```python 
x = [[4,7,2],[4,5,2],[9,4,2]]

ar = np.array(x)

ar.sum(axis=0) # this will print out the sum of axis 0 or all the rows 

# the output will be [17 16  6] this is the sum of axis 0 or all the rows in the matrix 
```

12. To find the `transpose` of an array 
```python 
Array1 = np.array([2,3,4],[3,4,2],[9,4,2])

print(ar.T) # this prints out the transpose
```

13. `Array1.flat` will give out the iterator then we can use for loop to print each element of the matrix 
```python 
for item in Array1.flat:
    print(item)
```

14. `ndim` gives out the dimension of the array 
```python 
print(Array1.ndim) # prints out dimension of array ie 2 in this case 
```

## Functions on Arrays 

15. `np.size` tells the number of elements in an array 
16. `np.nbytes` tells the size it takes inside the memory 
17. `np.argmax` will tell the location of the maximum element of the array 
18. `np.argmin` will tell us the index of minimum 
19. `np.argsort` gives out the order of the indices in which the array will get sorted 
20. `np.argmax(axis=0)` will give out the array in axis 0 (rows) which has maximum sum 
21. `np.sqrt(ar)` will give out a matrix with square root of each element 
22. `np.min` tells the minimum number in the matrix and `np.max` tells the maximum number in the matrix 
23. `np.where(ar>5)` tells us about the array with elements greater than 5 this will return a tuple 
24. `np.count_nonzero(ar)` will give you the number of non zero elements in the array 
25. `sys` is a python library that provides various functions and variable to manipulate various parts of python runtime environment 
```python
import sys 

pyarr = [0,3,5,2]
nparr = np.array(pyarr)
sys.getsizeof(1) * len(pyarr) # len returns the number of items of the objects 
```

26. `a.sum` to find the sum of arrays 
27. `a.mean()` to find the mean of the elements of the array 
28. `a.max()` to find the number of elements 
29. `a.cumsum()` cumulative sum of the elements 
30. `cumprod()` cumulative product of the elements of a matrix 
31. `a.var()` variance 
32. `a.std()` standard deviation 
33. To find the matrix multiplication of a matrix and a vector we can use the `@` function 
```python 
a = np.array((1,2))
b = np.array((2,2))

a @ b # this will give out the dot product 

a @ (0,1) # this will perform column vector multiplication 
```


You can access the [numpy library](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) and see all other methods and attributes for arrays 




