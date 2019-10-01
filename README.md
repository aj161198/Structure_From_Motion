# Structure_From_Motion
Implementation of Tomasi Kanade Factorization for sparse 3d reconstruction

Structure from motion is a photogrammetric range imaging technique for estimating three-dimensional structures from two-dimensional image sequences that may be coupled with local motion signals.
Main objective : To implement the research paper Shape and motion from image streams under orthography: a factorization method
https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/TomasiC_Shape%20and%20motion%20from%20image%20streams%20under%20orthography.pdf

Installation Instructions : - 
1. Run pip install -r requirements.txt (Python 2), or pip3 install -r requirements.txt (Python 3)
2. Download the dlib dat file from https://drive.google.com/open?id=1Drud1Z9dtxic05g7EZqJ3vEOxfkedcnJ 
3. Save it in this repository itself 
4. Run reconstruct.py using python3 reconstruct.py
5. It will ask path for images which you want to reconstruct, you can give your default path or test in on the sample_images repository.

Note : - The images are ordered in one particular direction
