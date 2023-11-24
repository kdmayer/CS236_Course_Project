### Running this repository

If someone else clones your repository, they need to initialize and update the submodules:

```
git clone git@github.com:kdmayer/CS236_Course_Project.git
cd CS236_Course_Project
git submodule update --init --recursive
```

The --recursive option is used to automatically initialize and update submodules during the cloning process.

### Adding a submodule

Use the following command to add a submodule to your repository:

```
git submodule add <repository_url> <subdirectory>
```

Replace <repository_url> with the URL of the GitHub repository you want to add as a submodule, and <subdirectory> with the path where you want the submodule to be placed within your repository.

For example:

```
git submodule add https://github.com/guochengqian/Magic123.git Magic123
```

After adding the submodule, you need to commit the changes to your repository:

```
git add .
git commit -m "Add submodule: repo"
git push
```

### Updating submodules

If the submodule repository is updated and you want to pull in the changes in your main repository, use the following commands:

```
git submodule update --remote
git add .
git commit -m "Update submodule to latest commit"
git push
```

### Google Cloud Platform

To identify available machines on Google Cloud Platform, run the following command:

```cloud compute machine-types list --filter="us-west1-b" | grep gpu```

### AWS

Navigate to your aws_key folder to connect to your VM with:

```ssh -i "cs236-key.pem" ubuntu@ec2-3-19-66-237.us-east-2.compute.amazonaws.com```