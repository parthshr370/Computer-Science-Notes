
# Git and Github

## Notes for working with git on command line/terminal 



1. **Git main** is the main branch in your local system
2. **Git origin** is the one on the browser online
3. `git init` to initialize git in the selected folder it is usually hidden in the directory as a dot(.) file
4. `git add .` will implement and register the change of a particular file to the staging area . we can say it will stage the file
5.  `git commit -m “the file has been changed”` is snapshot of your project, where a new version of that project is created in the current repository with a comment called “a file has been added”
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
38.  `git config user.email "partshr370@gmail.com”` to register your github username