# WCNN3D
A novel 3D Convolutional Neural Network (CNN) to perform organ tissue segmentation from volumetric 3D medical images.

For volumetric 3D medical image segmentation tasks, the effectiveness of conventional 2D CNNs are reduced due to loss of spatialinformation. The 3D CNN implements a cube-by-cube scanning strategy, followed by 3D transformation for each cube in terms of convolving and pooling. The 3D convolution is acquired in a hierarchical order starting from eliminating a 3D image into 2D slices. By replicating the convolution and pooling processes until the image informationis appropriately compressed, the CNN is ready for hierarchical learning from motifs to edges. By using 3D CNN, the image becomes scalable in the spatial direction, allowing accurate image detection with different frame sizes.

![Image text](https://github.com/arialibra/WCNN3D/blob/master/IMG-folder/conl.jpg)
( A conceptual 3D convolution layer which transforms consecutive 3D slices to feature spaces z1 and z2 )

3D OCT images with a resolution of 200 * 200 * 6 are used. Three levels of convolutions are calculated, with each level followed by a pooling process. After convolving and pooling, 32 feature volumes are formulated with only a limited number of pixels for learning. A fully connected hidden layer with 64 neurons is adopted to connect to the output layer and generate the prediction.

![Image text](https://github.com/arialibra/WCNN3D/blob/master/IMG-folder/CNN3D.PNG)
![Image text](https://github.com/arialibra/WCNN3D/blob/master/IMG-folder/2d.PNG)

The 3D CNN sufficiently considers the spatial scales of 3D images, and preserves the original imageinformation during the hierarchical representation process. A semantic OCT image segmentation model that adopts 3DCNN has shown promises for practical diagnosis and treatment planning, performing computation and communicationfunctionalities in a cyber-physical system for healthcare applications.

( Article: Lu, Hongya, et al. "A 3D Convolutional Neural Network for Volumetric Image Semantic Segmentation." Procedia Manufacturing 39 (2019): 422-428. )
