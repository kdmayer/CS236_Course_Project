# CS236_Course_Project

### Setup

After cloning the repository, set up the environment by executing

    conda env create -f cs236_project.yml
    conda activate cs236_course_project

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

    python3 train_gan_model.py -l sinkhorn -l energy -l gaussian -l laplacian --input_dir data
  
to run the code on your local machine and train the model for point cloud generation.

If you are on the sherlock cluster, you can run the code with the following command:

```bash
sbatch train_gan_model.sh
```

After training is complete, you can execute

    python3 test_gan_model.py

to test your code.

### Model Checkpoints

To copy back a trained model checkpoint from sherlock, run the following command from the project's root:

```bash
scp -r kdmayer@sherlock.stanford.edu:/home/groups/fischer/CS236_Course_Project/checkpoints/<checkpoint_name> checkpoints/
```

### Acknowledgements
 
This repository builds upon the Point Cloud GAN implementation: https://github.com/jacklyonlee/default-project

### Tips:

Convert .obj file to point cloud (here .las) with CloudCompare

```bash
open -a CloudCompare.app --args -O /users/kevin/desktop/example.obj -C_EXPORT_FMT LAS -SAMPLE_MESH POINTS 10000
```

Convert .ply point cloud file to mesh (here .obj) with CloudCompare (Throws a Triangulation error?!)

```bash
open -a CloudCompare.app --args -O /Users/kevin/Projects/CS236_Course_Project/mock_data/Interpolation_Output/0.ply -M_EXPORT_FMT OBJ -DELAUNAY 
```

Online viewer for .ply point cloud files at https://point.love/

Interesting package to look at: https://github.com/fwilliams/point-cloud-utils

 
