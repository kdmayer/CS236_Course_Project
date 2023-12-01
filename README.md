# CS236G_Course_Project

### Setup

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
pip install geomloss
```

### Usage Instructions

Execute

    python3 train_gan_model.py
  
to run the code in the repo and train the model for point cloud generation.

After training is complete, you can execute

    python3 test_gan_model.py

to test your code.

### Project Idea

- For this project, we will focus on modifying the code in trainer._train_step_g() to contain a number of different loss functions.
- Once the generator is trained, we will generate a number of point clouds and fit Axel's hyperplane model to them to visualize the generated output
- Lastly, we will explore the latent space of the generator by interpolating between points in the latent space and visualizing the intermediate point clouds
  
### Acknowledgements

The code builds on top of the provided default project. The configuration of parameters, e.g. the batch size, the 
number of epochs, or the loss function, has to be provided at the beginning of the train_gan_model.py and 
test_gan_model.py file, respectively. 

For reference, the original repo is available here: https://github.com/jacklyonlee/default-project

 
