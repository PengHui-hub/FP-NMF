# FP-NMF

This is the self-calibration algorithm for reconstructing redshift distributions using galaxy clustering.

The algorithm, inspired by the work of Zhang et al. ([2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...848...44Z/abstract)), has been developed in stages, incorporating the concepts of fixed-point iteration and Non-negative Matrix Factorization (NMF). These form the basis of two distinct parts of the algorithm, named as Algorithm 1 and Algorithm 2, respectively. An earlier version of the algorithm was introduced by Peng et al. ([2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.6210P/abstract)). The current algorithm, detailed in Peng et al. (2024), integrates the Nearly-NMF technique from Green and Bailey ([2024](https://ui.adsabs.harvard.edu/abs/2023arXiv231104855G/abstract)) to enhance NMF with new update rules, correctly accounting for nonuniform uncertainties and negative data elements.

## Dependencies

Core dependency:

   - numpy

## Contributions

Contributions and comments are always welcome.

