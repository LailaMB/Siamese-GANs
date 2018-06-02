# Siamese-GANs: Learning Invariant Representations for Aerial Vehicle Image Categorization 

## Overview
This is the implementation for the <b>Siamese-GANs</b> model architecture described in this paper: <a href="http://www.mdpi.com/2072-4292/10/2/351/htm"> "Siamese-GAN: Learning Invariant Representations for Aerial Vehicle Image Categorization"</a> by Laila Bashmal, Yakoub Bazi, Haikel AlHichri, Mohamad M. AlRahhal, Nassim Ammour and Naif Alajlan.

In this paper, we present a new algorithm for cross-domain classification in aerial vehicle images based on generative adversarial networks (GANs). The proposed method, called Siamese-GAN, learns invariant feature representations for both labeled and unlabeled images coming from two different domains. To this end, we train in an adversarial manner a Siamese encoder–decoder architecture coupled with a discriminator network. The encoder–decoder network has the task of matching the distributions of both domains in a shared space regularized by the reconstruction ability, while the discriminator seeks to distinguish between them. After this phase, we feed the resulting encoded labeled and unlabeled features to another network composed of two fully-connected layers for training and classification, respectively. 


## Prerequisites

To run the code, you need to install the following dependencies:
* <a href="https://www.tensorflow.org/"> Tensorflow </a>.
* <a href="https://keras.io"> Keras </a>. 
* NumPy.
* Matplotlib.
* Pandas_ml.
* Sklearn.

or use <code> requirements.txt</code> file: <br />
<code> pip install -r requirements.txt </code>

## Data
Datasets are not included in this project, as they are owned by other parties. However, you can prepare your own data or request the datasets used in the experiments from: 
* <b> Potsdam </b>, <b> Vaihingen </b>, and <b> Toronto </b> datasets are provided by <a href= "http://www.isprs.org/default.aspx"> ISPRS </a> and BSF Swissphoto. You can make a dataset request from <a href="http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html"> ISPRS Test Project on Urban Classification and 3D Building Reconstruction Request Test Data </a>. 

* <b> Trento </b> dataset is provided by Dr. Farid Melgani from the University of Trento.

## Feature extraction
<code> python MDPI_Feature_extractor.py </code>


## To run
<code> python MDPI_Scene_Siamese_GANs.py </code>


## Cite
If you find this code useful in your research, please, consider citing our paper:

<code> @Article{rs10020351,
AUTHOR = “Bashmal, Laila and Bazi, Yakoub and AlHichri, Haikel and AlRahhal, Mohamad M. and Ammour, Nassim and Alajlan, Naif”,
TITLE = “Siamese-GAN: Learning Invariant Representations for Aerial Vehicle Image Categorization”,
JOURNAL = “Remote Sensing”,
VOLUME = “10”,
YEAR = “2018”,
NUMBER = “2”,
ARTICLE NUMBER = “351”,
URL = “http://www.mdpi.com/2072-4292/10/2/351”,
ISSN = “2072-4292”,
DOI = “10.3390/rs10020351”
}
</code>  
  
