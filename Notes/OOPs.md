1. Object is a software bundle with certain state and behaviour . Used to model the real world objects that we find in everyday life.
2. Class is a **blueprint** or **prototype** which objects are created . It contains functions which are like blocs of code to execute some task. 
3. A **package** is a namespace that organizes a set of related classes and interfaces. Conceptually you can think of packages as being similar to different folders on your computer
4. **4 Pillars** of **OOPs** are  
  - Abstraction 
  - Encapsulation
  - Inheritance
  - Polymorphism

5.  **Abstraction** - Only the important details about an object are revealed . A car is viewed as a car rather than individual components like tire, window etc .
6. More like defining the required characteristics of an object and ignoring the irrelevant details.
7. Like Man driving a car wanting to increase the speed presses the accelerator but does not know that in pressing it the speed is actually increasing . He's unaware of the inner mechanism of the car 
8. **Encapsulation** - Wrappinig up the data under a single unit . It is the mechanism that binds together the code and the data it manipulates. Helps us to prevent data from being accesses from outside 
9. In encapsulation, the data in a class is hidden from other classes, which is similar to what ****data-hiding**** does. So, the terms “encapsulation” and “data-hiding” are used interchangeably.
9. Example of encapsulation
```java
class Employee {

    private int empid;
    
      private String ename;
}
```

10. **Inheritance** - Some objects have certain amount in common with each other Like mountain bikes , motor bikes all have **something common in them** . But they do have some properties that define that particular object 
   - **OOPs** allows us to inherit commonly used state behaviour from other classes. 
   - Like **Bicycle** becomes the *superclass* of Mountain bikes, Road bikes. In Java each class is allowed to have one direct *superclass* and unlimited *subclasses* 
11. To create a *subclass* use the word extends before your class definition 
```java
class MountainBike extends Bicycle {

    // new fields and methods defining 
    // a mountain bike would go here
}
```
Now MountainBike is a *subclass* inside a class called Bicycle

12. Interface is the blueprint of a class. Or we can just say Interface is a group of related methods with empty bodies 
13. Example of Bicycle Behaviour if specified as an interface 
```java 
interface Bicycle{
//wheel revolution per minute 

void changeCadence(int newValue);

void changeGear(int newValue);

void speedUp(int increment);
 
void applyBreaks(int decrement);
}
```
To implement this interface as the default interface we need to change the name of the object Bicycle eg Hero


14. The compiler will now require that methodschangeCadence, changeGear, speedUp, and applyBrakes all be implemented. Compilation will fail if those methods are missing from this class.

```java
class Hero **implements** Bicycle {

    int cadence = 0;
    int speed = 0;
    int gear = 1;
    
    void changeCadence(int newValue) {
         cadence = newValue;
    }

    void changeGear(int newValue) {
         gear = newValue;
    }

    void speedUp(int increment) {
         speed = speed + increment;   
    }

    void applyBrakes(int decrement) {
         speed = speed - decrement;
    }

    void printStates() {
         System.out.println("cadence:" +
             cadence + " speed:" + 
             speed + " gear:" + gear);
    }
}
```

15. **Polymorphism** is the ability of the programming language to diff between entities with same name by help of signature and declaration.
16. This class will contain **3 methods with same name**, yet the program will compile & run successfully
```java
public class Sum {
    // Overloaded sum().
    // This sum takes two int parameters
    public int sum(int x, int y)
    {
        return (x + y);
    }
  
    // Overloaded sum().
    // This sum takes three int parameters
    public int sum(int x, int y, int z)
    {
        return (x + y + z);
    }


    public double sum(double x, double y)
    {
        return (x + y);
    }
  
    // Driver code
    public static void main(String args[])
    {
        Sum s = new Sum();
        System.out.println(s.sum(10, 20));
        System.out.println(s.sum(10, 20, 30));
        System.out.println(s.sum(10.5, 20.5));
    }
}
```

17. **The static keyword:**  When we declares a class as static then it can be used without the use of an object in Java. If we are using static function or static variable then we can’t call that function or variable by using dot(.) or class object defying object oriented feature.
