This file just has information on how I tested code.

This time, I made most of my code in Google Colab so I could have access to a free GPU (Nvidia Tesla K80). You can see my work in it inside the Jupyter file 'lenet5.ipynb'.

Otherwise, you can just run 'parth_test.py' which will currently skip training and just load the best models from './trained_models' for img2obj and img2num. It will then display a picture of a cow and show that img2obj classifies it correctly in the window title. Once you hit a key, it will go away and starting using the webcam to classify images there.

You can uncomment my training and plotting code to generate the plots submitted in 'hw5.pdf' (the full writeup).
