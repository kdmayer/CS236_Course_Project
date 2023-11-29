## OUTDATED README

### Running this repository

If you clone this repository for the first time, you need to initialize and update the submodules:

``` bash
git clone https://github.com/kdmayer/CS236_Course_Project.git
cd CS236_Course_Project
git submodule update --init --recursive
```

The --recursive option is used to automatically initialize and update submodules during the cloning process.

### Adding a submodule

Use the following command to add a submodule to your repository:

```bash
git submodule add <repository_url> <subdirectory>
```

Replace <repository_url> with the URL of the GitHub repository you want to add as a submodule, and <subdirectory> with the path where you want the submodule to be placed within your repository.

For example:

```bash
git submodule add https://github.com/guochengqian/Magic123.git Magic123
```

After adding the submodule, you need to commit the changes to your repository:

```bash
git add .
git commit -m "Add submodule: repo"
git push
```

### Updating submodules

If the submodule repository is updated and you want to pull in the changes in your main repository, use the following commands:

```bash
git submodule update --remote
git add .
git commit -m "Update submodule to latest commit"
git push
```

## Tips from previous iterations of the project

### Installing threestudio on Sherlock

After cloning the repository as described above, run the following commands:

Request GPU resources: 

```bash
salloc -p gpu -G 1 --time=2:00:00 # --mem=64G --gres=gpu:1
``` 

Set up threestudio:

```bash
conda create --name 3studio python=3.9
conda activate 3studio
ml load py-pytorch/2.0.0_py39
pip install ninja
pip install -r requirements.txt
```

If your /home/users/kdmayer directory is full, check disk space usage with:

```bash
du -h --max-depth=1 /home/users/kdmayer | sort -h
```

If you need to clear disk space, try the following commands:
   
```bash
rm -r /home/users/kdmayer/.cache
rm -r /home/users/kdmayer/.local
```

### Connect to AWS

1. Open an SSH client.

2. Locate your private key file. The key used to launch this instance is cs236-key.pem

3. Run this command, if necessary, to ensure your key is not publicly viewable.

```bash
chmod 400 cs236-key.pem
```

4. Connect to your VM with:

```bash
ssh -i "cs236-key.pem" ubuntu@ec2-18-219-102-90.us-east-2.compute.amazonaws.com
```

### Google Cloud Platform

To identify available machines on Google Cloud Platform, run the following command:

```bash
cloud compute machine-types list --filter="us-west1-b" | grep gpu
```