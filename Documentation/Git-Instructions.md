# Instructions 

## Table of Contents
- [Clone Git Repo](#clone-git-repo)
- [Git-Commands](#git-commands)
  - [Typical Workflow](#typical-workflow)
  - [Fix errors](#fix-errors)
  - [Branches](#branches)
    - [Branch Workflow](#branch-workflow)
    - [Create branch](#create-branch)
    - [Switch branch](#switch-branch)
    - [Branch switch errors](#branch-switch-errors)
    - [Merging](#merging)
  - [General tips](#general-tips)

## Clone Git Repo

### Create SSH

#### Generate SSH key
To authenticate with GitHub via SSH, you need an SSH key pair. It can be created using the following command:
````
ssh-keygen -t ed25519
````

The command starts an interactive process:
1. **Storage Location:** The default (~/.ssh/id_ed25519) can be confirmed by pressing Enter.
2. **Passphrase:** Optionally, you can set a passphrase to further protect the key pair. If you don’t want one, simply press Enter.

The key pair will be stored in the ~/.ssh/ folder:
- `id_ed25519` – the **private** key
- `id_ed25519.pub` – the **public** key -> this will be uploaded to GitHub


#### Upload SSH key to GitHub

Before cloning the repository, the public key must be added to GitHub.

1. First, display the generated public key on the Raspberry Pi:
````
cat ~/.ssh/id_ed25519.pub
````
2. Copy the entire output. It will look something like this: ``ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... ``
3. Go to the [GitHub SSH-Settings](https://github.com/settings/keys)
4. Add the copied SSH key under SSH keys.


### Clone repository

To clone the repository, use the SSH URL from GitHub. You can find it under **Code** → **SSH**.

Once copied, run:
````
git clone SSH-URL
````

Then navigate into the project folder:
````
cd choice-prediction
````

## Git Commands

### Typical workflow

```text
Working directory → Staging area → Local repo → Remote (GitHub)
    (edit)            (git add)     (git commit)   (git push)
```

#### Pull latest changes from GitHub
```bash
git pull
```
- git pull fetches the latest changes from the remote repository and integrates them into your local project.
- **Important:** Always run git pull before making changes to stay up to date.

---

#### Show status

```bash
git status
```

- Shows modified files:
    - Green: already staged
    - Red: not yet staged

#### Add changes to staging area
```bash
git add file.txt   # Single file
git add .          # All changes
```

- The staging area acts as a buffer for changes before committing.


#### Save changes (commit)
```bash
git commit -m "Short description of the change"
```

- A commit is like a snapshot of your project.
- It saves staged changes (git add) and documents what was changed.

#### Upload changes to GitHub
```bash
git push
```

- git push uploads all local commits to the remote repository.

### Fix errors

Git errors can be fixed in different ways depending on **where in the workflow the mistake happened**:

#### 1. Discard changes in working directory (before git add)
```
git restore file.txt
``` 

#### 2. Remove changes from staging area (after git add)
```
git reset file.txt
```


### Branches

A branch is like a separate workspace where you can make changes and test ideas without affecting the main project.

- You can try out different ideas or simply experiment safely 
- Once finished, you can merge it back into the main branch

#### Branch Workflow

A common Git workflow looks like this:

```text
- main       # Production branch
- develop    # Development branch
- feature/*  # New features
- release/*  # Release preparation
- bugfix/*   # Bug fixes
- hotfix/*   # Urgent fixes

```


#### Create branch

````
git checkout -b new-branch-name
````

- Creates a new branch and switches to it immediately.
- **Important**: A branch is always based on the branch it was created from.


#### Switch branch
````
git checkout <branch-name>
````

##### Branch switch errors

Occasionally it happens that when you try to switch branches, you have made changes on your current branch that have not yet been added to the staging area. As long as you still have “uncommitted” changes on a branch, you cannot switch branches, and Git will display an error message.


Solutions:
1. Stage changes
    - Only if you are sure you want to keep them
3. Use git stash
    - ``git stash`` temporarily hides changes
    - Resets working directory to last commit
    - Can be restored later using git stash pop


#### Merging

````
git merge <branch-name>
````

- Combines/Merges one branch into another.
- **Important:** You must be on the target branch where changes should be applied.

### General tips

#### Git Stash

Useful when you:
- want to switch branches without committing changes
- want to move changes to another branch
- temporarily remove changes to test something else
````
git stash      # Temporarily save changes
git stash pop  # Restore and remove last stash
````

