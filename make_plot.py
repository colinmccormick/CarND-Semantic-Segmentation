# make_plot.py
# make plot of training losses for semantic segmentation

import matplotlib.pyplot as plt

# Vanilla FCN8. Epochs = 6, KP = 0.5, LR = 1e-3, Batch_size = 5.
loss_01 = [29.675,11.269,9.264,8.078,7.539,6.798]
# Now go to epochs = 24.
loss_02 = [47.963,12.867,10.149,9.292,8.826,8.066,6.946,6.651,5.844,6.010,5.119,7.305,11.876,7.744,6.766,5.740,5.718,5.078,7.041,6.918,5.891,4.686,4.171,4.255]
# Timing note: about 1:15 per epoch

# Now go to L/R flip augmentation with 6 epochs
loss_03 = [24.240,11.555,10.074,8.893,7.850,7.319]
# Now go to L/R flip augmentation with 24 epochs
loss_04 = [22.074,10.192,8.998,8.075,7.267,8.790,7.215,7.605,7.652,6.663,6.6166,7.362,6.382,4.971,7.031,5.365,5.271,5.306,7.625,4.809,4.396,4.240,3.906,3.825]

# Now correct L2 regularization by actually using it in the loss!
# L/R flip, 24 epochs, 5 batch size, KP = 0.5, LR = 1e-3
loss_05 = [44.9,22.6,19.7,18.4,17.2,16.4,15.2,15.1,14.9,14.3,12.9,12.7,12.1,11.5,11.1,10.7,10.5,10.1,9.7,9.8,9.6,9.1,9.3,9.4]
# Same, but multiply L2 by 0.1
loss_06 = [43.9,14.7,12.6,10.4,9.7,9.1,8.8,8.3,8.0,7.5,7.6,6.8,6.7,9.4,7.4,7.0,6.6,6.7,7.2,6.0,6.2,5.9,5.8,5.8]

e06 = [1,2,3,4,5,6]
e24 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

#plt.plot(e24,loss_02,'r-',e24,loss_04,'b-')
plt.plot(e24,loss_05,'k-',e24,loss_06,'g-')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()