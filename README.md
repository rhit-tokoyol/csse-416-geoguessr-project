# GeoGuessr Image Geolocation (CSSE 416)

This project guesses where a street-level photo was taken. It sorts each photo into one of 1,905 map areas (we call them cells) and guesses the center of the chosen cell. A guess is scored by how far it is from the true location, out of 5,000 points.

Everything for CLIP is in one notebook, main.ipynb. It loads the image list, runs every photo through a pretrained model called CLIP to turn it into a list of numbers, trains a small network on those numbers to pick the cell, and then tests it on photos it hasn't seen. 

To install what it needs, use Python 3.11 and run:

pip install -r requirements.txt

A GPU is strongly recommended. The first run downloads CLIP from Hugging Face, so it needs internet access.

The data is the OpenStreetView-5M dataset, about 5.1 million street-level photos from Mapillary (https://huggingface.co/datasets/osv5m/osv5m). We used the copy provided for the course. The data was extracted using import.ipynb. Used the same code, but changed the word 'test' to 'train' when we needed it.

To run it, open clip_model.ipynb and run all the cells, or run this in a terminal if in a SLURM cluster, sbatch train.sbatch.

For a quick check that takes a few minutes, set K = 2, n_samples = 20000 and num_epochs = 5 in the first cell. The full run uses K = None, n_samples = 1000000 and num_epochs = 20 and takes hours on a GPU. 

The results go into the output folder: curves.png shows how training went, a .pt file holds the trained network, and the cached photo features are saved so reruns are faster. The notebook prints the test score and accuracy at the end.

For running the ResNet101 model, open resnet_model.ipynb and run all cells (really slow), or run this in a terminal if in a SLURM cluster, sbatch train_resnet.sbatch

Note for the sbatch files:
You may have to adjust file locations to fit your system. (such as mkdir into local files systems for output results)
