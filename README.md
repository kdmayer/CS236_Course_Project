# CS236G_Course_Project

After cloning the repository, set up the environment by executing

    conda env create -f cs236_project.yml
    conda activate cs236g_course_project

If this doesn't work, make sure that you have conda 4.12.0/4.13.0 installed and install packages manually by testing 

```bash
python3 train_gan_model.py
```

and installing the missing packages with pip install <package_name>==<version>. On the sherlock cluster, this translates to the following steps:

```bash
conda create -n cs236 python=3.6
conda activate cs236
pip install torch==1.10.2 --no-cache-dir
pip install click
```

Then, execute

    python3 train_gan_model.py
  
to run the code in the repo and train the model for point cloud generation.

After training is complete, you can execute

    python3 test_gan_model.py

to test your code.
  
The code builds on top of the provided default project. The configuration of parameters, e.g. the batch size, the 
number of epochs, or the loss function, has to be provided at the beginning of the train_gan_model.py and 
test_gan_model.py file, respectively. 
