# Java Notes 
## ***With IntellijIdea Shortcuts***
1.  When variable type is defined before compiling and running the programme then it is called static programming language but when language allows you to modify and change the structures it is dynamic programming
2. Dynamic programming is for programmer efficiency but static is for hardware efficiency
3. Stack provides static memory storage while heap provides dynamic memory storage
4. In middle stage of java the file is converted into byte code and then to machine code
5. Java is platform independent 
6. JDK provides environment to develop and run the java program
7. It has all the tools like compiler, interpreter etc
8. The _Java Runtime Environment_, or _JRE_, is a software layer that runs on top of a computer’s operating system software and provides the _class libraries_ and other resources that a specific Java program needs to run.
9. Static variable can be used to refer to the common property of all objecrs (company name college name etc)
10. JVM has heap and stack memory allocation
11. Java source code --> JDK --> Bytecode --> JVM --> JRE
12. Make sure that the first letter is capital while defining a class (just a naming convention)
13. Public in public class means that class can be accessed from anywhere 
14. Main function the the block of code where the program starts or the entry point of the java program
15. `public static void main (String[] args)`  `public` − This is the access specifier that states that the method can be accesses publicly. `static` − Here, the object is not required to access static members. `void` − This states that the method doesn't return any value.
16. Main.class file gets created in the same location next to the java file
17. Main.class is the byte code that we run after compiling the code
18. To run a java code in terminal 
* Open terminal
* Open the location of the file by `cd` command
* Compile the code with javac(java compiler) command `javac hello.java`
* Now run the `.class` file that is created next to the original java file with `java` command `java Main.class`
19.   *public* is a Java keyword which **declares a member's access as public**. Public members are visible to all other classes.
20. Static variables and function that are not dependant on the objects .
22. A static method is **a method that belongs to a class rather than an instance of a class**. This means you can **call a static method without creating an object of the class**. Static methods are sometimes called class methods.
23. Return in programming means that whenever this function stops executing then it will give us some value 
24. *Void*  does not give any value .
25. File name and public class name should be same 
26.  . means current directory .. means previous directory 
27. in order to change the destination of the class file use the `-d` command . `javac -d .. Demo.java` will change the destination of the `.class ` file to the previous directory 
28. To tell the location of javac in the computer `where javac` 
29.  Type `psvm` in intellijidea to create` public static main void `
30. Type `sout`  in intellijidea to create `System.out.println`
31.  A package in Java is **used to group related classes**. Think of it as a folder in a file directory. We use packages to avoid name conflicts, and to write a better maintainable code.
32. Click on a particular class in java while holding ctrl key to get the information about that class .
33. the `println` means a new line or next line in java basic `System.out.print` will print without changing the line
34. ` System.out` is the standard output stream 
35. `System.in `is the standard input stream . The input we give from keyboard.
36. Every class in java extends to the object class 
37. `Scanner input = new Scanner(System.in);` is the command to initiate taking input from the standard input stream
38. `System.out.println(input.nextLine());` means that the output that is going to be printed is taken from the input we give from the input command and `nextLine` means that the whole line of string gets printed . Similarly `next` will simply print the first string till it finds space  . ` nextInt` will only print integers .
39. `input.nextInt()` will give you the next input in the form of integer 
40. The `nextInt()` method of a Scanner object reads in a string of digits (characters) and converts them into an int type.

### **Data Types with Typescripting**
41. **Primitive** is the most basic data type that cannot be broken into more data further .
42. **Float ,  int , char , boolean , long** are primitve data types 
43. Way to type the data types 
* `int rollno = 65 ;`
* `float num = 55.66f ;`
* `char letter = 'r' ;`
* `long verylong = 556789875434564344l ;`
* `double num = 567883.747467 ;`
44. `.` after a a variable gives you all the functions that can be applied to a datatype
45. `//` allows you to add comments to your java file 
46. In `System.in` the `in` variable is the keyboard .
47. `Scanner input = new Scanner(System.in);` The new keyword in Java **instantiates a class by allocating desired memory for an associated new object** 
**48. Programme to input the roll number given and** 
```java

import java.util.Scanner;  
  
public class Main {  
  
public static void main(String[] args) {  
  
Scanner input = new Scanner(System.in);  
  
int rollno = input.nextInt();  
  
System.out.println("this is your roll number " + rollno) ;  
  
}
```
```
```

49. The `nextInt()` method of a Scanner object **reads in a string of digits (characters) and converts them into an int type**. The Scanner object reads the characters one by one until it has collected those that are used for one integer
50.  **Code to print the input from the keyboard and print it**

```java

import java.util.Scanner;  
  
public class Main {  
  
public static void main(String[] args) {  

Scanner input = new Scanner(System.in);  

  
String Name = input.nextLine();  
  
System.out.println(Name);  
  
}  
  
}
```

51.  If we put the `next` instead of `nextLine()` then we just print the first input of the line
52. **Code to add two numbers**

```java
import java.util.Scanner;  
  
public class Main {  
  
public static void main(String[] args) {  

Scanner input = new Scanner(System.in);  
  
float num1 = input.nextFloat();  
float num2 = input.nextFloat();  
  
float sum = num1 + num2 ;  
  
System.out.println("sum = " + sum );  
  
}  
  
}
```
53. A first class object is _an entity that can be dynamically created, destroyed, passed to a function, returned as a value, and have all the rights as other_
54. We need to put the data type before each variable we create even while defining the relation between 2 other variables . Eg `float sum = num1 + num2 ` 
56. In Java, **type casting** is a method or process that converts a data type into another data type in both ways manually and automatically. The automatic conversion is done by the compiler and manual conversion performed by the programmer. In this section, we will discuss **type casting** and **its types** with proper examples.
57. Even when entering an integer inside a float we get a float . 
58.  To make this happen
* the destination type should be greater than the source type `float num = input.nextFloat();`  is possible because the destination is float which is greater than integer 
* `int num = input.nextFloat();` is not possible since the destination `int ` is smaller than the final type `float`

59. Typecasting in java is used to convert one data type to another . It is used to convert smaller type of data into larger ones 
60. Hierarchy of the datatypes **byte -> short -> char -> int -> long -> float -> double**
61. To convert float to integer . `int fun = (int)(67.7f);`
62. Automatic type promotions 

```java 
int a = 57;
byte b = (byte)a;

//if we put the value of byte more than 256 then the output we will get will be the remainder of 256 and the number we put . eg 257 will output 1.


```


63.  **Automatic type conversion**

```java 
byte a = 50 ;
byte b = 40 ;
byte c = 100

int d = (a*b)/c 

//the javac will automatically convert byte into a an integer as soon as it realises that the value of the expression reaches more than 256.
```

64.  Java follows unicode principles **Unicode is an international character encoding standard that provides a unique number for every character across languages and scripts making almost all characters accessible across platforms, programs, and devices.**

65. 
```java 

int number = 'A'

System.out.println("A");

//this will print out the askii value of A ie 65
```


### **Loops In Java** 

66. **for loop**

```java 
public class Basic {

public static void main(String[] args){


int a = 10'
if (a==10){

System.out.println("hello world ");

   }
  }
}

//this will print hello world since the condition is satisfied .

```

67. `.gitignore` tells Git about the files that you dont want git to track in your repository
68. A module file `.iml` is **used for keeping module configuration**. Modules allow you to combine several technologies and frameworks in one application. In IntelliJ IDEA, you can create several modules for a project and each of them can be responsible for its own framework.
69. Extensible Markup Language (XML) **lets you define and store data in a shareable manner**. XML supports information exchange between computer systems such as websites, databases, and third-party applications.
70. So count++ is the **post increment operator**. It will increment the variable but the expression will return the value of the variable before it was incremented.
71. Use of for loop
```java 
int a = 10;   
if (a == 10) {  
System.out.println("hello world this is true output");
```

72. **Using `while` loop**
 ```java 
 int count = 1;  
  
while(count != 5 ){  
  
System.out.println(count);  
count++ ;
```
It tells the system to increment the value of count till it reaches the condition where count is not equal to  5.

73. **Use of `for` loop in java**
```java 
for (int count = i ; count != 5 ; count++) {
System.out.println(count);
}
```
 this is exact same situation as we had for while loop where the output will print till 
 
74. **Program for Celsius to Fahrenheit calculator** 
```java
import java.util.Scanner;  
  
public class Main {  
  sout
public static void main(String[] args) {  
  
Scanner temp = new Scanner(System.in);  
System.out.println("enter the temperature in celcius ");  
  
float tempc;  
  
tempc = temp.nextFloat();  
  
float tempf = (tempc * 9/5 ) + 32 ;  
  
System.out.println(tempf);  
  
}  
  
}
```

75. **Java code to take input from the keyboard and then apply `if-else` loop** 
```java 
  public static void main(String[] args) {  
  
Scanner salary = new Scanner(System.in);  
  
System.out.println("enter the salary of the employee");  
  
int a = salary.nextInt();  
  
if(2000 < a){  
  
a = a + 10000;  
  
}  
else {  
a = a + 20000;  
  
}  
System.out.println(salary);  
  
}
```
Here you define that you need an input from the variable salary and then the input of salary that will be an integer will be taken up by variable ` int a = salary.nextInt()` 

76. In **IntellijIdea** the whole for loop can be created by typing `fori` 
77.  Shortcut `sout` is used for `System.out.println` to print in java
78. **While loop syntax**
```java
//while loop  
  
int a = 1 ;  
while(a<20){  
System.out.println("this is while loop");  
a = a + 3 ;  
  
}
```
79. You run `while` loop when you don't know how many times the loop is going to run and you run `for` loop when you know how many times the loop is going to run .
80. The body is executed at least once without checking the condition in `do-while` loop
81. **Code to print the maximum number of the 3 inputs**
```java 
System.out.println("enter 3 numbers");  
Scanner lar = new Scanner(System.in);  
  
int a = lar.nextInt();  
int b = lar.nextInt();  
int c = lar.nextInt();  
  
//find the largest of the 3 numbers  
int max = a ;  
if(b > max ){  
  
max = b; ;  
  
}  
if (c > max ){  
max = c ;  
  
  
  
}  
  
System.out.println("the largest number is " + max);  
}  
  
  
}
```

82. `for` loop Syntax
```java 
//For loop  

for (int i = 0; i < 20 ; i++) {  
  
System.out.println("hello world");  
  
  
}
```

83. The `trim()` method **removes whitespace from both ends of a string**. Note: This method does not change the original string.
84. The `charAt()` method **returns the character at the specified index in a string**. The index of the first character is 0, the second character is 1, and so on.
85. `System.out.println(word.charAt(0);` will take out the character from the input taken and then print the 
86. The use of `&&` and `||` . When `&&` is used , both the condition on either side needs to be satisfied but if `||` is used then any one condition true will make the whole true .
87. `Switch cases `is like a if else ladder that  helps us to execute a command with various conditions 
The syntax goes like 
```java 
// switch statement 
switch(expression)
{
   // case statements
   // values must be of same type of expression
   case value1 :
      // Statements
      break; // break is optional
   
   case value2 :
      // Statements
      break; // break is optional
   
   // We can have any number of case statements
   // below is default statement, used when none of the cases is true. 
   // No break is needed in the default case.
   default : 
      // Statements
}
```

Breaks are optional and the default condition is always executed no matter the input 

88. Put the cursor on the switch case and then press `ctrl+enter`  to convert the switch case into enhanced switch case 
89. An example of Switch case is given here

### Functions

A `function` is **a method that does have a return value**. To define a method to be a function, set its return type to be the type of the value it is returning.
A `method` in Java is **a block of code that, when called, performs specific actions mentioned in it**. For instance, if you have written instructions to draw a circle in the method, it will do that task. You can insert values or parameters into methods, and they will only be executed when called.

90. The `access modifiers` in Java specifies the accessibility or scope of a field, method, constructor, or class. We can change the access level of fields, constructors, methods, and class by applying the access modifier on it.
91. Access modifiers in Java are `Private , Default , Protected , Public `
92. Syntax of `methods` 
```java 
return_type name() {
//body
return statement;
}
```
93. The `void` keyword **specifies that a method should not have a return value**.
94. Static memory is the one that does not depend on the object in use .
95. In Java, the return keyword **returns a value from a method**. The method will return the value immediately when the keyword is encountered. This means that the method will not execute any more statements beyond the return keyword.
96. The type that you need to return is the same one you initiate 
```java 
// Start the method by string so the return value needs to be string 

static String name() {

String Greeting = " how are you " ;
return Greeting ;
}
```
97. Parameters **act as variables inside the method**. Parameters are specified after the method name, inside the parentheses. You can add as many parameters as you want, just separate them with a comma.
98. We can add the value of the of numbers when you are calling the method in main() bye the use of parameters inside `()`
```java 
int ans = sum3( a:30 , b : 20 ){
System.out.println(ans);
}
//this defines the two values of a and b inside parametres and then prints it 

static int sum3(int a , int b ){
int sum = a + b ;
return sum ;
}
```
98. You need to put the method() outside the `psvm` bracket . and then return the value in the end 
Code to return a string with parameters
```java 
public class Naming {  
  
public static void main(String[] args) {  
  
Scanner in = new Scanner(System.in);  //initiate the scanner command
System.out.println("Enter your Name : ");  //sout
String name = in.next();  //input string name using scanner 
String personalised = Greet(name);  //parameter for string
System.out.println(personalised);
}  
  //function initiation with static string 
static String Greet(String name){  
  
String message = "Hello there this is the way " + name;  
return message;  //returning the same datatype in the end 
  
  
}
```

99.  We put `static void methodname()` only when we know the value of the function wont return .
100. Code to pass the value of a string 

```java 

public class Passing {  
  
public static void main(String[] args) {  
  
String name = "Parth Sharma" ;  // this is assigning the name var a value that will eventually return 
  
greeting(name);  
  
  
}  
  
static void greeting(String name){  

System.out.println(name);
  
}  
}
```

101. **Java doesn't support pass-by-reference**. Primitive data types and Immutable class objects strictly follow pass-by-value; hence can be safely passed to functions without any risk of modification
102. When we pass a function in java even when the reference variables are different named they will still point towards the same object since they are a copy of the same object . 

example
```java 
public class Passing {  
  
public static void main(String[] args) {  
  
String name = "Parth Sharma" ;  
  
greeting(name);  
  
  
}  
  
static void greeting(namaskar){  

System.out.println(namaskar); // both namaskar and name are copy of the same object "Parth Sharma so they print the same value "
  
}  
}
```

Creating a new object inside the function 
here the new string `lmao` is only available in the scope of the function hence any action related to `lmao` can be only done inside the function . 

```java 
public class Passing {  
  
public static void main(String[] args) {  
  
String name = "Parth Sharma" ;  // name still points towards "Parth Sharma"
  
greeting(name);  

System.out.println(name); // even when you print name it will print "Parth Sharma"
  
}  
  
static void greeting(lmao){  

lmao = "Lakshay" // now lmao is pointing towards "Lakshay"

}  
}
```

103. Original Array `arr`  has a particular value . when the reference value is passed it will contain the copy of `arr` . Now the object is changed by `nums` then `arr `will also get changed . In The code above this point where there is creation of new string . There is new string formation because you cannot modify a string but rather create a string. In the code below we are able to modify the array because arrays are modifiable .
```java
import java.util.Arrays;  
  
public class Change {  
  
public static void main(String[] args) {  
  
int[] arr ={1,3,5,44,22,90}; // at first arr is pointing towards this array  
  
hello(arr);  
  
System.out.println(Arrays.toString(arr)); //printing the array  
  
}  
  
static void hello(int[] nums) {  
  
nums[0] = 99 ; //here the num now points towards arr and changes the value
  
}  
}
```
105. Method Overloading: Java allows defining multiple methods with the same name but different parameter lists. This is known as method overloading. Java determines which method to invoke based on the arguments provided.
```java 
public void methodName(int parameter) {
    // Code to be executed
}

public void methodName(String parameter) {
    // Code to be executed
}

```

104. The `java.util.Arrays.toString(int[])`method returns a string representation of the contents of the specified int array. The string representation consists of a list of the array's elements, enclosed in square brackets `("[]")` Adjacent elements are separated by the characters ", " (a comma followed by a space).
105. How to print an array in Java
```java
int[] arr ={1,3,5,44,22,90}; // at first arr is pointing towards this array  
  
hello(arr);  
  
System.out.println(Arrays.toString(arr)); //printing the array
```

106. A method wont act on a variable if it exists outside the var defining . You can only access the variables defined inside the function in the function.
```java 
public class Scope {  
  
public static void main(String[] args) {  
  
int a = 20 ;  
int b = 30 ;  
  
  
}  
  
static void random(){  
  
System.out.println(a);  // this wont work out since a is defined outside the methof random
  
}  
}
```

107. You cannot initialise the same reference value twice . its not possible to do `int a ` two times but you can change the value `a=45;` if needed .
108. Values initialised inside a bloc will remain in the bloc
```java 
  
{  
  
int z = 90 ;  
  
}  
  
System.out.println(z);  // cannot exist since the value z is initialised outside the bracket

```
109. Shadowing is when the variable defined in lower level bloc is defined and shadows the bloc of code in higher level . The value of lower level is taken into consideration and executed . 
```java 
public class Shadow {  
  
static int x = 90 ; // this is initialised inside this scope of public class  
//the value of larger scope gets shadowed when the var is initialised again  
  
public static void main(String[] args) {  
  
int x = 70;  // this one is executed 
  
System.out.println(x);  //prints out x = 70
  
}  
}
```

110. In order to return a value of a function in the form of `System.out.println()` we need to initialise Printstream library that is `import java.io.PrintStream;`
