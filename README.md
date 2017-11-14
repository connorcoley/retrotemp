# retrotemp

Neural network for predicting template relevance a la Segler and Waller's Neural Symbolic paper. 

Using 80%/10%/10% split on 5.4M Reaxys examples, the trained model at checkpoint ```ckpt-105660``` gets decent test accuracy. 43.1% top-1, 70.4% top-5, 78.3% top-10, 84.4% top-20, 89.8% top-50, and 92.6% top-100.

### Dependencies if you want to use the final model
- RDKit (most versions should be fine)
- numpy

### Dpendencies if you want to retrain on your own data
- RDKit (most versions should be fine)
- tensorflow (r0.12.0)
- h5py
- numpy
