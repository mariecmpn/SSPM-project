from fonctions.Tensor_Tucker import *
from Two_Pass import err2
from PIL import Image
image = Image.open('1200x768_le-panda-roux-est-.webp')
X = np.asarray(image)

#test One_Pass low recovery 

fig, axarr = plt.subplots(1, 4)
fig.set_size_inches(15, 4)

L = [[50, 20, 3], [100, 80, 3], [300, 200, 3], [600, 300, 3]]

for i in range(len(L)):
    X_hat_1pass,_ = One_Pass(X, L[i],2*np.array(L[i])+1 )
    X_hat_1pass = X_hat_1pass.astype('int8')
    image_stream = Image.fromarray(X_hat_1pass, 'RGB')
    axarr[i].imshow(image_stream)
    axarr[i].set_title( L[i])
plt.show()
