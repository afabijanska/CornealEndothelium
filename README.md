# Corneal Endothelium Image Segmentation

<p>This repopsitory contains a source code of algorithm for corneal endothelium image segmentation with U-Net based convolutional neural network. The source code may be used for non-commercial research provided you acknowledge the source by citing the following paper:</p>

<ul>
<li> <b>Fabijańska A.</b>: <i>Segmentation of Corneal Endothelium Images Using a U-Net-based Convolutional Neural Network</i>, Artificial Intelligence In Medicine, vol. 88, pp. 1-13, 2018, doi:10.1016/j.artmed.2018.04.004
</ul>

<pre><code>@article{Fabijanska2018,<br>
  author  = {Anna Fabija\'{n}ska}, <br>
  title   = {Segmentation of corneal endothelium images using a U-Net-based convolutional neural network},<br>
  journal = {Artificial Intelligence in Medicine},<br>
  volume  = {88},<br>
  number  = {},<br>
  pages   = {1-13},<br>
  year 	  = {2018},<br>
  note 	  = {},</br>
  issn 	  = {0933-3657},<br>
  doi 	  = {https://doi.org/10.1016/j.artmed.2018.04.004}, <br>
  url 	  = {https://www.sciencedirect.com/science/article/pii/S0933365718300575}<br>
}</code></pre>

# Running the code

## Prerequisites

Python 3.6, Tensorflow, Keras  

## Data organization

Organize your data as below. For training, keep the filenames consistent (an original image and the correspongind ground truth should be files of the same name saved in diifferent locations).

<pre><code>
├───project_dir<br>
    └───data<br>                    # data directory
        └───test<br>                # test images
        |   └───org<br>             # original images of corneal endothelium
        |   |   testFile1.png <br>
        |   |   testFile2.png <br>
        |   |   testFile3.png <br>
        |   └───preds<br>           # images predicted by the network
        └───train               # train images
            └───org<br>             # original images of corneal endothelium
            |   trainFile1.png <br>
            |   trainFile2.png <br>
            |   trainFile3.png <br>
            └───bw<br>              # ground truths (black = 0, white = 255)
                trainFile1.png <br>
                trainFile2.png <br>
                trainFile3.png <br>
</code></pre>

## Repository content

<ul>
  <li> <b>configuration.txt</b> - file to be edited; contains data paths and train/test setings 
  <li> <b>prepare_train_set.py</b> - script for extracting random patches from train images and saving them as hdf5 files (to be run first)
  <li> <b>training.py</b> - script for training U-Net with patches loaded from hdf5 files (to be run second)
<li> <b>predict.py</b> - script for performing image segmentation; segmentation is performed for all images from an indicated directory (to be run third)
  <li> <b>helpers.py</b> - some helper functions for reading/writing data
</ul>

# Contact

<b>Anna Fabijańska</b> <br>
Institute of Applied Computer Science <br>
Lodz University of Technology <br>
e-mail: anna.fabijanska@p.lodz.pl <br>
WWW: http://an-fab.kis.p.lodz.pl
