# CPU 

1. First CPU was **intel 4004** a 4 bit *CPU* 
2. **Assembly** is a helpful syntax for reading and writing machine code that’s easier for humans to read and write than raw bits
![Assembly](https://cpu.land/images/assembly-to-machine-code-translation.png)

3. RAM is your computer’s main memory bank, a **large multi-purpose space** which stores all the data used by programs running on your computer. That includes the program code itself as well as **the code at the core of the operating system**. The *CPU always reads machine code directly from RAM*, and code can’t be run if it isn’t loaded into RAM.
4. CPU has a **instruction pointer** in it which points to a location in the **RAM** to fetch the next instruction . After the instruction is executed the CPU pointer moves the pointer and repeats the cycle 

![instruction pointer](https://cpu.land/images/fetch-execute-cycle.png)

After the completion of the task the pointer moves immediately to the new instruction in the RAM
5. This instruction pointer is stored in a **register** .Registers are small storage buckets that are extremely fast for the CPU to read and write to

6. At the end of this process there’s machine code in a file somewhere. The operating system loads the file into RAM and instructs the CPU to jump the instruction pointer to that position in RAM. The CPU *continues running its fetch-execute cycle as usual*, so the program begins executing.

7. When you boot up your computer the instruction pointer starts at a programs somewhere that program is the **kernel** . Kernel has the full access to the computer memory, peripheral and other resources.
  
8. **Windows kernel** is called ***NT Kernel***

9. Modern architecture have 2 modes to access the kernel , **Kernel mode** and the **User mode** .

  
10. In kernel mode any instruction in the supported instruction is allowed . In user mode only a subset of instructions are allowed. Generally kernels and drivers run in kernel mode while application runs in user mode,
![modes](https://cpu.land/images/kernel-mode-vs-user-mode.png)

11. Current privilege level (CPL) contains two least significant bits of the cs register. Two bits can store four possible rings . Rings 1 and 2 are designed to run drivers.

15. Syscall or sytem call is **a way for programs to interact with the operating system**. A computer program makes a system call when it makes a request to the operating system's kernel.
16. It provides a layer of security for user to not misuse the kernel 
17. If you’ve ever written code that interacts with the OS, you’ll probably recognize functions like `open`, `read`, `fork`, and `exit`. Below a couple of layers of abstraction, these functions all use _system calls_ to ask the OS for help.
18. Programs need to pass data to the operating system when triggering a syscall; the OS needs to know which specific system call to execute alongside any data the syscall itself needs.
19. This is where **API**(Application programming interface) comes in . SInce it is widely immpractical for user to use the syscall for every program so we use the api .
