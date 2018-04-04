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

- Numpy
  - [Arrays](http://cs231n.github.io/python-numpy-tutorial/#numpy-arrays)
  - [Array indexing](http://cs231n.github.io/python-numpy-tutorial/#numpy-array-indexing)
  - [Datatypes](http://cs231n.github.io/python-numpy-tutorial/#numpy-datatypes)
  - [Array math](http://cs231n.github.io/python-numpy-tutorial/#numpy-math)
  - [Broadcasting](http://cs231n.github.io/python-numpy-tutorial/#numpy-broadcasting)

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

详细用法参考 [classes](https://docs.python.org/3.5/tutorial/classes.html)。