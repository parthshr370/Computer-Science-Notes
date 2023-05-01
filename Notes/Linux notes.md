# Linux terminal commands

### All commands required while using linux terminal and some more like *omzsh* and *vim* to simplify terminal usage


_**COMMAND LINE CODES**_

1.  `CTRL + ALT + T` for opening the terminal window
2.  `Echo` hello world for printing any text in the terminal
3.  `Cal` - type this to get the calendar of the month
4.  `Cal 2017` will give the whole calander for 2017
5.  `Cal - y` for the calendar of current year
6.  `Date` for getting date time and timezone
7.  `Clear` command for clearing the whole terminal
8.  `History` for getting all our previous commands
9.  `! (the number )` eg !1 to get to th first command of the history
10.  `!!` for the most recent command
11.  `history - c ; history - w` clear the history and makes the changes permanent
12.  `exit` for closing the teminal
13.  `CTRL + SHIFT +M` for bringing back menu bar
14.  `which` command is used to tell the location the particular command eg which cal will give you `/usr/bin/cal`
15.  `$path` will tell you the whole path to reach particular directory
16.  `man` command tells you about the manual of linux wil eg `man -k which` will open manual then to expand the manual type `man 1` which will open manual page for which command
17.  `lf` gives you name of home directories `man lf` will give description page for `lf`
18.  `cat` is used to print the standard output of stream
19.  `ctrl + d` to exit the `cat` command
20.  `date > date.txt` will print the output of date command in date.txt file
21.  `.> is for input and .< for output`
22.  `cut` command is to cut a specific section in the file
23.  `date | cut — delimiter” ” — field=1` will cut out first field in the date command
24.  .`|` is used for piping arguments
25.  `echo` does not accept standard input like `date` but you can use `xarg`
26.  `date | xargs echo` xarg will pipe up the standard command
27.  `rm` command will remove something eg `rm delete.txt`
28.  `alias getdate = “some longass command in between ”` will assign the alias the long command
29.  `(/)root` directory is the upper most one which is like the admin in windows
30.  `cd` to change directory (this command is case sensetive)
31.  `mkdir` is for creating a new folder in the specific directory.`mkdir hello` will just create a folder named hello in the home directory
32.   `ls -R` will display all the files in sub directory
33.  `ls -a` will list all the directory in the system
34.  `echo “hello world” > file.txt` to print an input into the text file
35.  piping is output of the first command is used as input of 2nd one `cat file.txt | tr a-z A-Z > upper.txt`
36.  `Touch text/houses.txt` will create a folder named text and then a text file names houses in it
37.  `mv` is used to move a file from one location to other `mv name.txt random` will move the text file to a folder named random
38.  `cp -R` test home to copy file from test to home directory and create a new folder named home with the contents of test
39.  `df` command displays the amount of disk space left in the directory
40.  `tail hello.txt` will print the log in the file
41.  `tail -n2` will print the last 2 lines in the files
42.  `head` will print the logs from above
43.  `diff` command helps you to compare two files line by line
44.  `locate “filename”` to find the location of the file in the system
45.  `find . -type f -name “two.txt”` to find f here is for file with the specified name
46.  `find . -type f -name “*.txt”` will find random files with .txt files
47.  `read` `write` and `execute` are the 3 permissions in system
48.  `chmod` command sets the permission and directory of the file
49.  `chmod [reference][operator][mode] file...` is the format for chmod command
50.  `chown root` gives root permission to a file
51.  `sudo chown root upper.txt` will give upper.txt root permissions
52.  `ctrl+s` to suspend output in terminal and `ctrl+q` to resume output
53.  `find . -type f -name “*.txt” -exec rm -rf {} +` to execute removing all the files
54.  `grep` searches a file for a particular pattern of characters, and displays all lines that contain that pattern.
55.  `grep “hello” unit.txt` will search hello in the file unit.txt
56.  `grep -c “parth” mine.txt` will tell the count of parth in mine.txt
57.  `grep -p “\\d{3}-\\d{3}-\\d{3}” company.txt` will print you the digits in the specified format in the file
58.  `— version` after any command gives you the current version of the installed program
59.  `ctrl+k` in linux terminal will remove all the text in front of the cursor
60.  `top` to see everything running on the pc
61.  `hostname` to tell the name of the host
62.  `lscpu` to find the detail of cpu
63.  `uname -o` will give you GNU/Linux
64.  `nano [filename.sh](<http://filename.sh>)` will create a bash script for you
65.  `#!` is called shabang in bash script it is used to start bash scripts
66.  anything below `#!/bin/bash` is interpreted as a normal command
67.  `mkdir ~/Desktop/helloworld` to create
68.  `touch file{1..100 }` will create a 100 folders in the directory
69.  `cd ~/Desktop/helloworld` to change directory to helloworld directory
70.  `ls -lh ~/Desktop/helloworld > ~/Desktop/magic.log` to print the log file
71.  `htop` command is an advanced version of top command
72.  [`https://ohmyz.sh`](https://ohmyz.sh/) for zsh on steroids
73.  `kate hello.c` will open a c file in the kate text editor
74.  `type gcc hello.c` to compile the file and then type `./a.out` to run the command in terminal
75.  `g++` for running c codes in linux with mathematical operations
76.  `lsblk` is **used to display details about block devices** and these block devices.
77.  `lspci` is a command on Unix-like operating systems that prints ("lists") detailed information about all PCI buses and devices in the system
78. `.` means the current directory `..` means the previous directory
78.  Notes on using vim in linux terminal

-   `vim hello.c` will create the file and then vim opens up in the terminal
-   `vim filename.txt` to open a text file in vim text editor
-   `:w` to save the file
-   `:q` to quit the file
-   `:q!` to quit the file without saving
-   `:wq` to save and quit the file
-   `:set nu` to show line numbers
-   `:set nonu` to remove line numbers
-   `:noh` to remove highlighting of search terms
-   `:syntax on` to enable syntax highlighting
-   `u` to undo the last command
-   `ctrl + r` to redo the last command
-   `/text` to search for the specified text
-   `dd` to delete the current line
-   `yy` to copy the current line
-   `p` to paste the copied line
-   `:s/old/new/g` to replace old text with new text
-   `:%s/old/new/g` to replace old text with new text in the entire document