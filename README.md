# Subpixellic Methods for Sidelobes Suppression and Strong Targets 

This project is an implementation of the first method of the following paper : *Subpixellic Methods for Sidelobes Suppression and Strong Targets*, Abergel, R., Denis, L., Ladjal, S., & Tupin, F. (2018).


## Running the tests

Composition:
-  *irregular_translations.py*: it contains the functions needed to perform the TV minimization
- *regularization.py*: it contains the functions needed to perform the Regularization of the translations maps (Chambolle-Pock, Weights computation)
- *image_operators.py*: it contains the functions needed to perform the Chambolle-Pock algorithm 
- *tests.ipynb*: it contains all the tests and the figures. 


## Acknowledgement
[1] CPSAR-TOOLS : Complex Pseudo-RAW SAR Image Processing Toolbox, Rémy Abergel, Loïc Denis, Saïd Ladjal and Florence Tupin, http://helios.mi.parisdescartes.fr/~rabergel/

[2] A first-order primal-dual algorithm for convex problems with applications to imaging, Antonin Chambolle and Thomas Pock 

[3] FFTW package, Matteo Frigo and Steven G. Johnson, http://www.fftw.org/ 

[4] https://github.com/jyhmiinlin/pynufft

[5] Matthew Ferrara (2020). NUFFT, NFFT, USFFT (https://www.mathworks.com/matlabcentral/fileexchange/25135-nufft-nfft-usfft), MATLAB Central File Exchange. Retrieved April 9, 2020

[6] https://github.com/Rareform/Chambolle-Pock







