
# **Python Notes**
1. In python the code gets executed line by line 
2. Directly gets executed without use of a compiler like C/C++ just type `python hello.py` on the terminal to run the code .
3. Platform independent language . Easily portable 
4. Since the interpreter prints the code line by line so when we 
```python
price = 10
price = 20
print(price)
```

### Data types 

```python
price = 20 #integer 
rate = 4.9 #this is float 
light_status = true # boolean
name = 'zenitsu' # string

```
6. Boolean is case sensitive type `True` `False`  will only work 
7. Input taking in python takes place with `input` command , we take a variable then equate it to input and put a text inside the input bracket 
```python
name = input('hello please type in your name')

print('Welcome to this world ' + name)
```
8. Since we cannot add or subtract diffrernt data types without typecasting so 
```python 
age = input('Enter your age ')

newage = 2023 - age 
print(newage)
```
This will not work since we will probably try to subtract string from integer 
The correct way will be 
```python
age = input('Enter your age')
newage = 2023 - int(age)
print(newage)
```
9. Python type() is a built-in function that **returns the type of the objects/data elements stored in any data type or returns a new type object depending on the arguments passed to the function**. The Python type() function prints what type of data structures are used to store the data elements in a program.
```python 
age = input('Enter your age')
print(type(age))
newage = 2023 - int(age)
print(type(age))
print(newage)
```
Now in this case the output will be standard output but with `class 'str'` and `class 'int'` being string and integer classes each.
11. Type casting is **a method used to change the variables/ values declared in a certain data type into a different data type to match the operation required to be performed by the code snippet**.
12. In the input function **you always get a string**  but if you expect it to change to some other data type we use typecasting
13. Task - **Convert weight in kg to in pound**
```python
pound = input("what is your weight ")
kg = int(pound) * 0.453
print(kg)
```
Here we convert the default string value in input to that of integer and then there is automatic typecasting inside the python code converting it into float 

14. To create multi line quotes we use triple quotes and feel the data in between 
```python 
''' Hello how are you elliot alderson 
oh shit im talking to you ?
  this is not working 
  '''
```
15. Python strings are indexed just like arrays the first character is [0] and then [1]  [2] 
```python 
char = 'Hellopycharm'

print(char[0]) # this will print up the character at 0 place that is H
```
16. We can also put negative characters in the [ ] square brackets which will represent the characters from right side 
```python 
char = 'What happens when'

print(char[0]) # this will print the 0th char that is W 
print(char[-2]) # this will print the 2nd last character ie e 
print(char[1:3]) # this prints all the char from 1 to 3 
print(char[0: ]) # this will start from 0th place in the index and move upto the end of the index
print(char[ : 5]) # python will automatically assume 0th index before : 

```
17. Formatted strings are useful when you want to dynamically generate text with variable 
```python
# our goal is lenon [smith] is a coder
name = 'lenon'
last = 'john'
message = name + '[  ' + last + ' ] is a coder '
```
This will work perfectly fine but will get complex as we add more codes so ...
18. A formatted string starts with an f in the start and then single quotes 
19. `stringname = f'{name} + [{last}] is a coder '` the curly brackets are the string placeholders so this helps us to hold our strings in the formatted string 

### Methods in Strings


20. To calculate the length of the string we can use a built in function called length `len` 
```python 
elliot = 'Hello friend'
print(len(elliot))
```
The output will be the number of boxes in the index
21. intname followed by a dot will open up all the functions for a string like uppercase function 
22. print and len are basic functions while functions like upper() are string specific so they are called methods
```python
elliot = 'Hello friend'
print(elliot.upper())  # output is HELLO FRIEND
```
this will not modify the string but rather create a new string  `HELLO FRIEND`
23. `find()` method will locate the index of a particular character in a string
```python 
hello = 'Namaste'
print(hello.find('N')) # the output will be 0
```
If we pass a string parameter that does not exist in the string then the output will be -1

24. `replace()` function will replace the specified string to the one desired 
```python 
elliot = 'Hello friend'
print(elliot.replace(friend,dost)) # output will be Hello dost
```
25. `in` method is used to check the boolean if the value is in the string or not
```python
hello = 'namaste london'
print('london' in hello)
```
the output for the code will be true since `london` exist in the string `hello`
26. **Summary of string specific methods**
```python 
course = 'Python for cobras'
len()                     #length of string
course.find()             #find the index of string 
course.upper()            #
course.replace()
'...' in course 
```

### Arithmetic


27. There are various arithmetic expressions . Here is the list of the ones listed down below
```python 
print(10 / 3) #this will print 3.3333(float)
print(10 // 3)  #this will print 3 (int)
print(10 % 3)  #this will print the remainder 1
print(10 ** 3)  # this is exponent output is 10000
```

Some useful maths functions in python
```python
x = 2.9 
print(round(x)) # this will round off to 3 
print(abs(-2.9))  # this is absolute function which will always give +ve value 

```

28. We can import maths functions into python file by using `import maths` in the start of the file 
29. You can open the [math module in python documentations](https://docs.python.org/3/library/math.html) to learn how to use various maths functions
### Conditional loops


30.  When you write any condition and press enter after the colon `:` the text editor automatically indents the code for that function 
31. To return to the normal indentation we use `shift + tab` 
32. We use conditions such as `if` `elif` and `else` 
```python
is_hot = False
is_cold = True

if is_hot:

   print('It is a very hot day')

   initial = 100000

credit_good = True


if credit_good :
    Downpay = 0.1*initial

else : 
    Downpay = 0.2*initial
    
print(f"payment down is  :  {Downpay}")print('Get in your ac')

elif is_cold:

   print("how are you buddy")

else:

   print('How are you')

   print('Go touch some grass')
```

33. For `and` to be true you need to satisfy both the conditions and for `or` to be true any one can be true to give output 
34. We use conditional operators like `> < =` in python 
```python 
temprature = 30

if temprature > 30 :
  |
  print("Its a hot day ")
 
else:
 | 
  print("Go touch some grass")
```

Code for Length name of the string 
```python 
name = 'Just'

if len(name) > 3 :
 |
 print("this is too short")

elif len(name) < 50 :
 |
 print("Name more than 50 ")

else :
 |
 print("Soja bhai")
```

```python
weight = int(input('Weight: ')) # typecasting from string to int 
unit = input("Pound(L) or Kilo(K) ")

if unit.upper() == 'L' : # the input will be uppercase 
 |
 converted = weight*0.45 
 |
 print(f"Youre weighing pretty heavy in {converted} pound")

else:
 |
 converted = weight
 print(f"the weight of the fat guy is {weight} Kg ")
```

35. Use of `while` statement 
```python 
i = 1

while i <= 5:
 |
 print('*' * i ) # i into * number of times
 i = i + 1
 |
 print("Done") #this will print i pattern till it reaches 5 stars
```

Guessing Game using while for loops
```python 
secret_number = 9 # the number that will cause break of the loop
guess_count = 0
guess_limit = 3

while guess_count < guess_limit : # the limit for number of inputs
 |
 guess = int(input('Guess the number : '))
 guess_count += 1 # increase 1 till 3rd iteration
 if guess == secret_number :  # if 9 then 
 print('You won! ')
 break #break the loop
```

36. A `for` loop is used for iterating over a sequence (that is either a list, a tuple, a dictionary, a set, or a string .
```python
for item in 'Python' : # here item is the var and Python is the string
 |
 print(item) # this will print Python one by one each iteration
```
37. Lists are used to store multiple items in a single variable. More like arrays in Java . enclosed inside [ square brackets ] 
```python
for item in ['Parth' , 'Shivansh' , 'Mannan '] :
 |
 print(item)

#this will print the items in the list one by one
```

38. The `range()` function returns a sequence of numbers, starting from 0 by default, and increments by 1 (by default), and stops before a specified number.
39. If we give the range parameters like `range(5,10)` our output will start from 5 and terminate at 10
40. In `range()` the first two parameters give you the start and end point of the range while the 3rd parameter tells you about how many steps to take in between each iteration .
41. `range(2,11,4)` the output will start from 2 and then take 4 steps when till it reaches 11 . If the end point is before the next step then we only get the output of iteration before the final one .
42. We have nested loops in python for `for` loop and `while` 
```python 
#syntax for nested loops
Outer_loop Expression:

    Inner_loop Expression:

        Statement inside inner_loop

    Statement inside Outer_loop
```

![nested loops](https://media.geeksforgeeks.org/wp-content/uploads/20220801153940/Nestedloop.png)

```python
for x in range(4): # if x single handedly will print 0,1,2,3
    for y in range(3): # we tell that y holds value of 0,1,2
     print(f"({x},{y})") # this will print all possible values of x,y


# this will print all the x and y coordinates which we defined in for nested loop

The output will be 

(0,0)  
(0,1)
(0,2)
(1,0)
(1,1)
(1,2)
(2,0)
(2,1)
(2,2)
(3,0)
(3,1)
(3,2)
```
Here in the first iteration of the outer loop `x` is `0` and in the inner loop `y` is `0` . Now for 2nd iteration the control goes back to line `2` or the one with `y`  where y is `1` then `2` then the loop gets back to line `1` 

43. Multi line comments are not possible in python
```python
numbers = [5,2,5,2,2]

for int in numbers : 
        output = '4' #initiate a var nameed output and initiate to empty string
        for count in range(int) : # range of the 
            output += 'X' # incriment the value each time
        print(output) # print 


The output will be a line of 5 charaters starting with 4 and then pattern of X 

4XXXXX
4XX
4XXXXX
4XX
4XX


```
Basically what is happening in this `nested loop` is when we initiate for loop for `int` in the list `number` .If we just print this statement we will get the counting of all elements in `numbers` .But here first we initialise an empty string in the loop then we initiate another for loop that will create a variable `count` and take values and iterate values from the `range()` of `int` variable that is what each iteration in int value gives . Now that the `count` var takes value from `range(int)` in the loop and prints output with `output` being `output = output + 'X'` . 

Inner loop takes `range(int)` first as 5 then prints the value of 5 X  then goes back to the main loop then takes the value of 2 , then prints 2 X .

### Lists and Tuples

44. **Python Lists** are just like dynamically sized arrays, declared in other languages (vector in C++ and ArrayList in Java). In simple language, a list is a collection of things, enclosed in [ ] and separated by commas. 
```python
names = ['josh','lennon','gwen','miles']

print(names) # this will print all the items in the lists [josh,lennon,gwen,miles]

print(names[0]) # this will just print josh
print(names[0:2])  # this will print item at index 0 and 2


# we can also modify the list

names[0] = 'joshua'
 # this will update the list and replace the string at 0 with joshua
```

Code to print the maximum number in the list 
```python
numbers = [3,5,6,2,10]

max = numbers[0] # we just initialise max and assume that index 0 is the largest number

for new in numbers:
    if new > max: # if the value of any number in numbers list is greater than max then max is equal to the number
        max = new
print(max)
```

45. A **2D array** is basically a matrix or rectangular array of numbers 
```python 
matrix = [

   [2,5,7]  #1
   [4,6,9]  #2 
   [5,5,5]  #3
]

matrix[0] # this will return another list which is #1

```
This is a matrix or a 2-D List where each item in our outer list is another list

46. Printing output of characters inside matrix
```python
matrix = [

   [2,5,7], #1
   [4,6,9], #2 
   [5,5,5]  #3
]

for row in matrix:
    for item in row:
        print(item)
        
```
Now here we can see that we start a for loop where `row` in `matrix` will return just the 3 different lists in python with given values now we initiate another for loop where the `items` in the `row` will be returned . this will first happen for 1st list then 2nd then 3rd 
This way the output will be elements in the matrix 

47. **Operations we can perform on the list** 
```python 
numbers = [3,6,4,8,4]

number.append(20) # this will add 20 to the end of the list
number.insert(2,10) # the first number is the number insert after in the list
# the 2nd number is the number we need to insert 
number.remove(4) # will remove 4 from the list 
number.clear()  # this will clear the whole list with nothing but squares bracket
number.pop() # removes the last item in the list 
number.index(6) # returns the index position of particular value 
number.count(4) # this will return the number of iterations of 4 in the list ie2
number.sort()
print(number) # this will print [3,4,4,6,6]
# we can also reverse the list by 
number.reverse()

number2 = number.copy() # this will create a copy of original number list
```


48. Program to remove duplicates in a list 
```python
number = [4,7,2,3,3]

unique = [] # intialise with empty list

for new in number : # start for loop to iterate value of number in new var 
    if unique not in new : # finding the values not in unique 
        unique.append(new) # and then appending/adding the values (no duplicates formed)
print(unique)
```

49. **Tuples** are similar to **list** but we cannot modify them we cannot add new items , they are immutable . Tuple uses (`circular brackets `) instead of square brackets in lists .
50. We only have two methods in **tuples** , count and index .
```python
coordinates = (2,4,6) # this is a touple 
```
51. Unpacking in Python refers to **an operation that consists of assigning an iterable of values to a tuple (or list ) of variables in a single assignment statement**.
```python 
coordinates = (2,4,5)
x,y,x = coordinates # this assigns x as index [0] y as index [1] and z as [2]
print(y) # this will print 4
```
52. **Dictionaries** are used to store data values in key:value pairs . A **dictionary** is a collection which is ordered, changeable and do not allow duplicates. 
53. Dictionary is written in curly brackets and has keys and values.
54. Key is the name we assign to the item in dictionary . And values lie after `=`
```python
costumer = {
			"name" : "Homelander" ,
			"age" : 30,
			"Is aadhar" : True  

print(costumer["name"]) # tHis gives value inside name key
}
```

55. We can also supply value and key by using `dictionary_name.get()` method.
```python 
costumer = {
			"name" : "Homelander" ,
			"age" : 30,
			"Is aadhar" : True  

print(costumer.get("birthdate","Jan 1 1999") # creates a ley named birthday and value jan1 
}
```

Creating a digit mapper that converts number to text
```python
phone = input("Enter the phone number")

digit_maps = {
    
    "1" : "One",
    "2" : "two",
    "3" : "Three",
    "4" : "Four"
}
output = "" # initiate an empty output string
for ch in phone :
    output += digit_maps.get(ch,"!") # character is set if nothing is given 
    print(output)
```

56. The `split()` function is used to split a string items in a list into a list 
```python
text = "welcome to jungle"

x = text.split()

print(x)

#output is ['welcome', 'to', 'jungle']
```

### Functions

57. **Python Functions** is a block of statements that return the specific task.
58. We initiate a function in python by using `def` keyword then the `function_name()` and `:`
![Python function](https://media.geeksforgeeks.org/wp-content/uploads/20220721172423/51.png)

```python 
def greeting() :
  | print("hello there")
  | print("dont code plz")
greeting()
```
this executes the function `greeting()`

59. Parameters in functions are used to pass information in the function 
60. Parameters are the place holders for receiving information
61. The parameter acts like a local variable inside the function that calls the 
62. When we have a parameter we are obligated to pass a value through it .
63. `parameters` are the holes or placeholders that we define in a function `arguments` are the actual pieces of information we put inside the function .
64. We can also put more than one parameters inside the function()
```python
def greetings(name,surname):
    print(f"hello how are you {name}  {surname}")
    print("Please print something")
    
greetings("John","sharma")

# we can also do 

greetings(name= "john",last = "sharma")
```
65. Return statement is used to return values to the caller of our function
```python
def square():
 |return number*number

result = square(3) #passing value as 3 and finding result 
print(result)
```

66. By default all the functions return the value `none`
67. We use `try except` to handle errors with invalid datatypes inputting

68. `ValueError` is given when unknown data type is entered 
69. `ZeroDivisionError`  when we type 0 and the value is not possible 
```python
try:
    age = int(input("Enter age"))
    income = 2000
    risk = income/age
    print(age)
except ValueError :
    print("Incorrect Datatype entered")
except ZeroDivisionError :
    print("value cannot be zero")
```

### Classes
70. Classes are used to define new types in python. We have basic types like `Numbers` `Strings` `boolean` . Then we have some complex types like `lists` `Dictionary`
71. We define classes by `class` keyword followed by the name in capital letters 
72. Creating a new class creates **a new type of object**, allowing new instances of that type to be made. Each class instance can have attributes attached to it for maintaining its state. Class instances can also have methods (defined by their class) for modifying their state.
73. **Syntax:** Class Definition
```python
class ClassName:
    # Statement
```
**Syntax:** Object Definition
```python
obj = ClassName()
print(obj.atrr)
```

74. An object is an instance of a class . Class defines the blueprint of an object .
75. To create object we type the name of our class then store that object in a variable .
76. We then type out our object and then a `.(dot)` to start calling the class we created .
```python
class Pointing :
    def meth(self):
        print("meth")
    def coke(self):
        print("hellofriend")
        
point = Pointing() # defines the object 

point.meth() #initiates the class in the object 
```

77. Objects can also have attributes .Attributes are variables that belong to a particular object 
```python
point = Pointing() # defines the object
point.x = 10 # tells us the value of x in point object
point.y = 20 # tells us the value of y in point
point.meth() #initiates the class in the object
```
