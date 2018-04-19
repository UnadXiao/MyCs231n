# Python / Numpy Tutorial

This tutorial was contributed by [Justin Johnson](http://cs.stanford.edu/people/jcjohns/).

如果有Matlab基础的话，推荐使用这个教程 [numpy for Matlab users](http://wiki.scipy.org/NumPy_for_Matlab_Users) 。

[CS228](https://cs.stanford.edu/~ermon/cs228/index.html)课程也有[Python Tutorial](https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb)。

## Table of contents

- [Python](#python)
  - [Basic data types](#basic data types)
  - [Containers](#containers)
    - [Lists](#lists)
    - [Dictionaries](#dictionaries)
    - [Sets](#sets)
    - [Tuples](#tuples)
  - [Functions](#functions)
  - [Classes](#classes)

- [Numpy](#Numpy)
  - [Arrays](#arrays)
  - [Array indexing](#array-indexing)
  - [Datatypes](#datatypes)
  - [Array math](#array-math)
  - [Broadcasting](#broadcasting)

 - SciPy

  - [Image operations](http://cs231n.github.io/python-numpy-tutorial/#scipy-image)
  - [MATLAB files](http://cs231n.github.io/python-numpy-tutorial/#scipy-matlab)
  - [Distance between points](http://cs231n.github.io/python-numpy-tutorial/#scipy-dist)

 - Matplotlib

   - [Plotting](http://cs231n.github.io/python-numpy-tutorial/#matplotlib-plotting)
   - [Subplots](http://cs231n.github.io/python-numpy-tutorial/#matplotlib-subplots)
   - [Images](http://cs231n.github.io/python-numpy-tutorial/#matplotlib-images)

## Python

Python是高级，动态类型，多泛型的编程语言。下面是一个经典快速排序算法的Python代码。

```python
def quicksort(arr):
	if len(arr) <= 1:
		return arr
	pivot = arr[len(arr) // 2]
	left = [x for x in arr if x < pivot]		# 列表推导，速度比for快
	middle = [x for x in arr if x == pivot]
	right = [x for x in arr if x > pivot]
	return quicksort(left) + middle + quicksort(right)
print(quicksort([3, 6, 8, 10, 1, 2, 1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"
```

### Python versions

Python存在2.7和3.5+两个版本，同时这两个版本互相不兼容。本课程使用的是Python3.5版本。在命令行输入命令`python --version`检测你的Python版本。

## Basic data types

Python的基础类型有整形，浮点，布尔和字符串。

### Numbers:

```python
x = 3
print(type(x)) # Prints "<class 'int'>"
print(x)       # Prints "3"
print(x + 1)   # Addition; prints "4"
print(x - 1)   # Subtraction; prints "2"
print(x * 2)   # Multiplication; prints "6"
print(x ** 2)  # Exponentiation; prints "9"
x += 1
print(x)  # Prints "4"
x *= 2
print(x)  # Prints "8"
y = 2.5
print(type(y)) # Prints "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"
```

注意：Python没有`x++,x--`自增/减操作。

Python对于复数有自己的内建类型，具体内容参考 [numeric-types-int-float-complex](https://docs.python.org/3.5/library/stdtypes.html#numeric-types-int-float-complex)。

### Booleans

Python的逻辑运算使用的是英文，而不是常见的`&&`和`||`。布尔类型只有两个值`True`和`False`。

```python
t = True
f = False
print(type(t)) # Prints "<class 'bool'>"
print(t and f) # Logical AND; prints "False"
print(t or f)  # Logical OR; prints "True"
print(not t)   # Logical NOT; prints "False"
print(t != f)  # Logical XOR; prints "True"
```

### Strings

Python对字符串的支持很强大，有许多好用的函数。

```python
hello = 'hello'    # String literals can use single quotes
world = "world"    # or double quotes; it does not matter.
print(hello)       # Prints "hello"
print(len(hello))  # String length; prints "5"
hw = hello + ' ' + world  # String concatenation
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"
```

```python
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"
```

其他的python字符串函数可以参考[string-methods](https://docs.python.org/3.5/library/stdtypes.html#string-methods)。

## Containers

Python包含很多种内建容器类型：**list**，**dictionaries**，**sets**和**tuples**。

### Lists

Lists可以看作是可变长度的数组，并且可以存储不同类型数据。

```python
xs = [3, 1, 2]    # Create a list
print(xs, xs[2])  # Prints "[3, 1, 2] 2"
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
xs[2] = 'foo'     # Lists can contain elements of different types
print(xs)         # Prints "[3, 1, 'foo']"
xs.append('bar')  # Add a new element to the end of the list
print(xs)         # Prints "[3, 1, 'foo', 'bar']"
x = xs.pop()      # Remove and return the last element of the list
print(x, xs)      # Prints "bar [3, 1, 'foo']"
```

更多细节参考 [more on lists](https://docs.python.org/3.5/tutorial/datastructures.html#more-on-lists)。

**Slicing:** 除了一次访问一个元素之外，Python还提供一个简洁的语法访问sublists。这就是切片(slicing)。

```python
nums = list(range(5))     # range is a built-in function that creates a list of integers
print(nums)               # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(nums[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # Assign a new sublist to a slice
print(nums)               # Prints "[0, 1, 8, 9, 4]"
```

**Loops:** 我们可以按照下面方式遍历一个list元素。

```python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# Prints "cat", "dog", "monkey", each on its own line.
```

如果想要获取每个元素的索引，可以使用内建函数`enumrate`

```python
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line
```

**List comprehensions:** <u>列表推导(List comprehensions)是在C环境下运行的，所以速度比for循环快。</u>程序上常常需要将数据从一种类型转换到另外一种类型，例如计算一个list各个元素的平方。

```python
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # Prints [0, 1, 4, 9, 16]
```

使用列表推导方式简化代码：

```python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # Prints [0, 1, 4, 9, 16]
```

甚至添加条件：

```python
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"
```

### Dictionaries

字典是存储键值对(key, value)的容器。

```python
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'     # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del d['fish']         # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```

详细内容参见[dict](https://docs.python.org/3.5/library/stdtypes.html#dict)。

**Loops  : ** 字典循环方式如下

```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

可以使用`items`方法，同时获取键和值。

```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

**Dictionary comprehensions : **和[列表推导](#Lists)类似，可以很方便的构建一个字典。

```python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"
```

### Sets

set是一组无序唯一的元素集合。

```python
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
animals.add('fish')       # Add an element to a set
print('fish' in animals)  # Prints "True"
print(len(animals))       # Number of elements in a set; prints "3"
animals.add('cat')        # Adding an element that is already in the set does nothing
print(len(animals))       # Prints "3"
animals.remove('cat')     # Remove an element from a set
print(len(animals))       # Prints "2"
```

详细用法参考 [set](https://docs.python.org/3.5/library/stdtypes.html#set)。

**Loops : **可以使用和List循环类似的方法循环Set元素。有一段值得注意，Set是无序的所以无法知Set中元素的顺序。

**Set comprehensions :** 构建方式同List/Set推导。

```python
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # Prints "{0, 1, 2, 3, 4, 5}"
```

### Tuples

Tuple和List很像，Tuple是**有序不可变**的List。最大的不同点是：Tuple可以作为字典([Dict](#dictionaries))的键，集合([Set](#sets))的元素，而List不可以。

```python
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"
```

详细用法参考 [tuples](https://docs.python.org/3.5/tutorial/datastructures.html#tuples-and-sequences)。

## Functions

Python使用关键字`def`来定义函数：

```python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
# Prints "negative", "zero", "positive"
```

定义可选参数的函数：

```python
def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

hello('Bob') # Prints "Hello, Bob"
hello('Fred', loud=True)  # Prints "HELLO, FRED!"
```

详细用法参考[defining-functions](https://docs.python.org/3.5/tutorial/controlflow.html#defining-functions)。

## Classes

Python中定义类的方式很直截了当。

```python
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```

###### 详细用法参考 [classes](https://docs.python.org/3.5/tutorial/classes.html)。

## Numpy

Numpy是科学计算的核心库，它提供了高性能多维度数组的对象和处理这些数组的工具。如果你熟悉MATLAB你可参考这个[教程](http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users)。

### Arrays

numpy数组是矩阵排列的数值，使用非负下标来访问。numpy数组的维度是数组的*rank*，数组的*shape*是数组各个维度的元组大小。

可以使用Python的lists来初始化numpy，通过"[ ]"访问各个元素：

```python
import numpy as np
a = np.array([1, 2, 3])		# Create a rank 1 array
print(type(a))				# Print "<class 'numpy.ndarray'>"
print(a.shape)				# Print "(3, )"
print(a[0], a[1], a[2])		# Print "1 2 3"
a[0]=5						# Change an element of the array
print(a)					# Prints "[5, 2, 3]"

b = np.array([1, 2, 3], [4, 5, 6])		# Create a rank 2 array
print(b.shape)							# Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])		# Prints "1 2 4"
```

Numpy提供了很多种创建array的函数：

```python
import numpy as np

a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
```

你可以从[arrays-creation](https://docs.scipy.org/doc/numpy/user/basics.creation.html#arrays-creation)获取更多创建array的方法。

### Array indexing

Numpy提供了多种方式访问array。

**Slicing :**和list类似，array也可以切片。多维的array需要指定各个维度上的切片。

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
# array的切片和原始数据指向的是相同数据，改变切片数据同样会改变原始的数据
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"
```

您也可以将整数索引与切片索引混合在一起。但是，这样做会产生比原始数组更低级别的数组。注意，这与MATLAB处理数组切片的方式完全不同：

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
```

**Integer array indexing :**当使用整数下标对numpy数组切片时，得到的数组视图将始终是原始数组的子数组。 相比之下，整数数组索引允许您使用另一个数组的数据构造任意数组。 这里是一个例子：

```python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"
```

整数数组索引的一个有用技巧是从矩阵的每一行中选择或改变一个元素：

```python
import numpy as np

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])
```

**Boolean array indexing :**布尔数组索引可以让你可以挑出数组中的任意元素。通常，这种索引类型用来选择满足一些条件的数组元素。下面是例子：

```python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"
```

为了简洁起见，我们忽略了大量有关numpy数组索引的细节; 如果你想知道更多，你应该阅读文档[arrays-indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)。

### Datatypes

每个numpy数组都是相同类型的元素的网格。 Numpy提供了一组可用于构造数组的数字数据类型。 Numpy在创建数组时尝试猜测数据类型，但构造数组的函数通常还包含可选参数以明确指定数据类型。 这里是一个例子：

```python
import numpy as np

x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # Prints "int64"
```

关于numpy datatypes的内容可以阅读[array-dtype](https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html)。

### Array math

基本的数学函数在数组上按元素运算，并且可以作为运算符重载和numpy模块中的函数使用：

```python
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
```

请注意，与MATLAB不同，`*`是元素乘法，而不是矩阵乘法。 我们使用`dot`函数来计算向量的内积，将向量乘以矩阵，并乘以矩阵。 `dot`可以作为numpy模块中的一个函数，也可以作为数组对象的一个实例方法：

```python
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
```

Numpy为数组执行计算提供了许多有用的函数; 其中最有用的是`sum`：

```python
import numpy as np

x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
```

您可以在文档中找到由numpy提供的数学函数的完整列表[math](https://docs.scipy.org/doc/numpy/reference/routines.math.html)。

除了使用数组计算数学函数外，我们还经常需要矩阵变维(reshape)或以其他方式处理数组中的数据。 这种类型的操作最简单的例子是转置一个矩阵; 要转置矩阵，只需使用数组对象的`T`属性即可：

```python
import numpy as np

x = np.array([[1,2], [3,4]])
print(x)    # Prints "[[1 2]
            #          [3 4]]"
print(x.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v)    # Prints "[1 2 3]"
print(v.T)  # Prints "[1 2 3]"
```

Numpy提供了很多处理数组的函数，参见[array-manipulation](https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)。

### Broadcasting

广播是一种强大的机制，允许numpy在执行算术运算时与不同形状的数组一起工作。 通常我们有一个更小的数组和更大的数组，我们希望多次使用更小的数组来对更大的数组执行一些操作。
例如，假设我们想要为矩阵的每一行添加一个常量向量。 我们可以这样做：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)
```

这有效; 但是，当矩阵`x`非常大时，在Python中计算显式循环可能会很慢。 请注意，将向量`v`添加到矩阵`x`的每一行相当于通过垂直堆叠`v`的多个副本来形成矩阵`vv`，然后执行`x`和`vv`的元素求和。 我们可以像这样实施这种方法：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

Numpy广播允许我们执行此计算，而不实际创建v的多个副本。考虑使用广播的此版本：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

由于广播机制，即使`x`大小是`（4,3）` 、`v`大小是`(3, )`，`y = x + v`这行公代码也起作用。这行代码就好像`v`实际上已经变维`（4，3）`了一样，其中每一行都是`v`的一个副本，并且求和是按照元素进行的。

广播两个阵列遵循以下规则：

1. 如果数组不具有相同的等级，则用1来预先给出较低等级数组的形状，直到两个形状具有相同的长度。
2. 如果这两个数组在维度中具有相同的大小，或者如果其中一个数组在该维度中具有大小1，则这两个数组被称为在维度中兼容。
3. 如果阵列在所有维度上兼容，阵列可以一起广播。
4. 广播后，每个阵列的行为就好像它的形状等于两个输入阵列的形状的元素最大值。
5. 在一个数组的大小为1而另一个数组的大小大于1的任何维中，第一个数组的行为就好像它是沿着该维复制的。

如果这种解释不理解，请尝试阅读文档[broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)或此解释[EricsBroadcastingDoc](http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc)中的解释。

支持广播的功能被称为通用函数(*universal functions*)。 您可以在文档[ufuncs](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs)中找到所有通用函数的列表。

以下是广播的一些应用：

```python
import numpy as np

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(x + v)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((x.T + w).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)
```

广播通常会让你的代码更加简洁快捷，所以你应该尽可能地使用它。

### Numpy Documentation

这个简短的概述触及了你需要知道的有关numpy的许多重要事情，但还远远没有结束。 查看numpy的参考资料[numpy reference](https://docs.scipy.org/doc/numpy/reference/)，了解更多关于numpy的信息。