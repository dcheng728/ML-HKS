# Mesh Learning

This repository includes the code to adversarially train a contrastive neural network on 3D meshes. The performance of the contrastive neural neural network is benchmarked by comparing its clustering results to a classifier based on heat kernel signature. 

The following is the file structure of the code:

```bash
├── SHREC11_plus
|   (a directory to hold the dataset with noise)
├── ContraNN+atkNN.ipynb
├── Transformation_Experiment.ipynb
├── Classifier.ipynb
├── weights
|   (saved weights from training)
├── helpers
|   (various helper methods)
│   ├── preproc.ipynb
│   ├── NNs.ipynb
│   ├── misc.ipynb
│   ├── compare.ipynb
└── .gitignore
```

Davidson: the file structure above is just a draft of what the final directory can look like.