
# Git and Github

## Notes for working with git on command line/terminal 



1. **Git main** is the main branch in your local system
2. **Git origin** is the one on the browser online
3. `git init` to initialize git in the selected folder it is usually hidden in the directory as a dot(.) file
4. `git add .` will implement and register the change of a particular file to the staging area . we can say it will stage the file
5.  `git commit -m “the file has been changed”` is snapshot of your project, where a new version of that project is created in the current repository with a comment called “a file has been added”

6. The **-m** in `commit -m` is the message you want to display along with the message 

6.  `git status` tells us the status of the current repository , if there is any branch to commit or anything else.
7.  `git restore —staged names.txt` will remove the file from the staging area and restore the original file
8.  `git log` will tell you the logs of all the actions that took place in the file . all the commits with specific hashid
9.  `git reset <hashid>` will remove all the commits before the commit of given hashid . or will just unstage the files from the area
10.  `git stash` temporarily shelves (or stashes) changes you've made to your working copy so you can work on something else, and then come back and re-apply them later on.
11.  you need to do `git add .`to send the file to staging area each time
12.  `git stash pop` will take content from the stash file and bring it to the staging area
13.  `git stash clear` will clear the changes made in the backstage
14.  `git remote add origin <url of your repository you just created>` creates a remote connection of the cloned repository
15.  use of `origin` is name of the url of the repository
16.  `git remote -v` will show all the url attached to the folder
17.  `git push` pushes the commits from the local repository to the remote repository
18.  `git pull` pulls the commits from the remote repository to the local repository
19.  `Branches` allow you to develop features, fix bugs, or safely experiment with new ideas in a contained area of your repository
20.  `git branch` creates a new branch in the repository
21.  `ls .git` will list the elements of the master branch
22.  `git merge feature` will add the committed/checked out branch to the main branch.
23.  `Forking` a project is copying a project into your own account
24.  `git clone <the url>` will create a local download of the repository
25.  The `git remote` command lets you create, view, and delete connections to other repositories
26.  `upstream` is the orginal repository from which the code is taken
27.  `git remote add upstream` will add the origin file aka the `upstream` file into the remote area that is the online repo
28.  `pull request` allows you to make changes in your own personal branch to the main branch
29.  `git pull` will only work in the origin branch because it is the one that you create and unable to upstream to the main branch
30.  one branch can only open one pull request
31.  `git reset` will reset all the commits just next line below it
32.  `git add .` will just make the staging area updated of the changes
33.  `git fetch - - all - - prune` will fetch all the commits in the upstream even the ones that were pruned(removed)
34.  `git pull upstream main` will pull all the code from main branch to your remote branch to sync with the branches
35.  squash will convert all the commits into the commit previous to them pick is the exact opposite
36.  `git checkout <branch name>` will switch the branch
37.  `git merge <branch name>` merges the branch with the current working branch
38.  `git config user.email "<youremail@xyz.com>”` to register your github username

39. Whenever i tried to push the file by using `git push origin master ` you will be asked your username and password. It might show errors like `remote: Support for password authentication was removed on August 13, 2021.`
to solve this 
   * Go to `settings` --> `developer settings` --> `Generate Token(classic)` --> Generate a token by `ticking all the boxes` 

     * Make sure to **copy the token code** and then put the copied token code in the password section while pushing the code 


 To create a new file and add it to origin 

```
echo "# new" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:parthshr370/new.gitgit push -u origin main
```

To add an existing repository

```
git remote add origin git@github.com:parthshr370/new.gitgit branch -M main
git push -u origin main

```

The use of `-u` is to **upload**  

So the last line of the code says that push and upload the file to the main branch of the origin file


40.  *"The tip of your current branch is behind its remote counterpart"* means that there have been changes on the remote branch that you don’t have locally. And Git tells you to import new changes from **REMOTE** and merge it with your code and then push it to remote.

You can use this command to force changes to the server with the **local repository** (). remote repo code will be replaced with your local repo code.

`git push -f origin master`

With the `-f` tag you will override the remote branch code with your local repo code.

41. `git whatchanged --since "1 week"` will give you all the changes made to repository in the timeframe

## How to create a Pull Request 

1. A way to submit you contribution to another devs repository 
2. Go to the github repo you want to contribute to 
3. **Fork the repo**  into your very own github account 
4. **Clone your fork** into your local machine 
5. Open the project into the code editor 
6. Use `git branch < branchName >` to create a branch 
7. Use **git checkout** to move to the new branch 
8. After making the changes use `git add `to put the file in the staging area 
9. Then use`git commit -m "displaymessage"`
10. Push the commit by using `git push origin < branch name >` 
11. You can now go to github and start a **pull request** 
12. Make sure you follow the contribution guidelines before creating a pull request 