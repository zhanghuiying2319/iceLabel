import numpy as np
from matplotlib.pyplot import cm
import matplotlib.image as mpimg

with open('../data/withoutBoundaries_multiLabel_Huiying_relabelSmallsamples_newIce0.5V2_180222.npy','rb') as f:
    a = np.load(f,allow_pickle=True)#img
    b = np.load(f,allow_pickle=True)#size
    c = np.load(f,allow_pickle=True)#label

habit = []
for idx, (img, size, label) in enumerate(zip(a, b, c)):
    file_name = str(idx) + '.jpg'
    label = np.array([label[:7].argmax(), label[7], label[8], label[9]])

    label = ' '.join(str(j) for j in label)
    habit.append(file_name + ' ' + label)

    mpimg.imsave('../data/imgs/' + file_name, img, cmap=cm.gray)

#plt.imshow(a[0], plt.cm.gray)

with open('../data/label.txt','w') as f:
    for l in habit:
        f.writelines(l+'\n')
