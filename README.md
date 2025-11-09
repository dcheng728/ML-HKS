# 3D point cloud learning

This project explores how to extract meaningful and robust representations from point cloud data using **graph neural networks (GNNs)**. An **adversarial neural network** is trained alongside the GNN to generate small perturbations, forcing the GNN to learn features that are stable under noise and deformation. The model’s learned features are evaluated against the **heat kernel signature (HKS)**, demonstrating that the method provides a scalable alternative to computationally expensive geometric descriptors.

# Motivation

Point cloud is commonly encountered in modern applications (e.g. self-driving cars). The meaning of a point cloud lies in the shape, something humans can easily recognize but not for computers.
It is therefore desirable to store data in a form from which a computer can easily extract meaning. This goal defines **representation learning**, a branch of machine learning focused on discovering informative and discriminative features from raw data.

![image.png](3D%20point%20cloud%20learning/image.png)

# Methods

### Graph Neural Networks (GNNs)

Point clouds represent geometry through the spatial distribution of points. Their meaning does not depend on point order, rather on the relative positions of points, making **graph neural networks (GNNs)** well suited for encoding them. GNNs learn structure by exchanging information among neighboring nodes.

### Robustness and Adversarial Training

Recently, robustness to adversarial perturbations has received increasing attention across machine learning applications. These perturbations are small, carefully crafted changes that are imperceptible to humans but can drastically alter a model’s output.

The same concern applies here. By using a graph neural network (GNN), we inherently encode certain symmetries that raw coordinate data lack. However, we also seek robustness to additional perturbations, such as sensor noise and measurement errors, to ensure that our model remains stable under such variations.

![An example of how a ML algorithm may be sensitive to a perturbative when it shouldn’t.
Yao Z. and Gao J., “Adversarial Example Defense Based on the Supervision,” IJCNN 2021](3D%20point%20cloud%20learning/Screenshot_2025-11-06_at_4.12.23_PM.png)

An example of how a ML algorithm may be sensitive to a perturbative when it shouldn’t.
Yao Z. and Gao J., “Adversarial Example Defense Based on the Supervision,” IJCNN 2021

To address this, we also build an **adversarial neural network**, that learns to apply small perturbations. The adversarial neural network learns to generate small input perturbations that strongly affect the main model’s representation. The two models are optimized jointly in an adversarial framework to enhance robustness.

![Screenshot 2025-11-06 at 4.38.44 PM.png](3D%20point%20cloud%20learning/Screenshot_2025-11-06_at_4.38.44_PM.png)

### The Loss Function

The quality of a learned feature can be judged by how well it preserves similarity between objects. If two objects belong to the same category, their feature representations, even when derived from different point clouds, should lie close together in feature space. In contrast, features of objects from different categories should be far apart.

$$
Loss = 
(\text{same class distance}) - (\text{different class distance})+
\text{adversarial}
$$

# Results

We trained on the [SHREC11](https://arxiv.org/abs/1102.4258) dataset , which contains 600 point clouds from 30 different classes. 

The following plot is a histogram of the distances between same-class point clouds, and different-class point clouds, before and after the perturbations by the adversarial neural network.

![image.png](3D%20point%20cloud%20learning/image%201.png)

Below the histogram, we also visualize the learned features by projecting them onto their three largest principal components obtained from PCA.

After the adversarial neural network applies its perturbations, the feature distribution becomes more dispersed. However, this dispersion is not sufficient to cause significant overlap between point clouds of different classes. This indicates that the adversarial training has been effective in making the main GNN robust to such perturbations.

### Benchmark (HKS) Comparison

Alternatively, features can be extracted from point cloud data using the **heat kernel signature (HKS)**. The HKS models heat diffusion on a surface, describing how heat spreads from each point over time. This captures both local and global geometry and is invariant to isometric deformations. However, computing HKS requires solving the eigenvalue problem of the Laplace-Beltrami operator, which scales poorly with the number of points and is costly for large datasets.

We compare the two approaches by clustering their learned features and treating the clusters as classes in a multi-class task. Across 30 classes, our GNN (10 nodes per layer) achieved 43.5% accuracy, while HKS reached 51.6%.

![image.png](3D%20point%20cloud%20learning/image%202.png)

# Run the Code

- Presentation.ipynb: This file was ran live May 2023 at Colorado College, it outlines the principles behind the GNN setup and adversarial training procedure within the notebook in a logical way to an audience of undergraduate Math and CS students.
- ContraNN+atkNN.ipynb: the main file that trains the performs the training loop where the GNN is trained against the adversary
- L.ipynb: the loss function