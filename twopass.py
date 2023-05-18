from fonctions.Tensor_Tucker import *

from PIL import Image
image = Image.open('1200x768_le-panda-roux-est-.webp')
X = np.asarray(image)



# Test Two-pass on low_recovery 

fig, axarr = plt.subplots(1, 4)
fig.set_size_inches(15, 4)
plt.rc('text', usetex=True)

L = [[100, 300, 3],[300, 500, 3], [600, 800, 3],[800, 900, 3]]
#Try with two pass 
for i in range(len(L)):
    X_hat_2pass,_,_ = Two_Pass(X, L[i])
    X_hat_2pass = X_hat_2pass.astype('int8') # compulsory 
    image_stream = Image.fromarray(X_hat_2pass, 'RGB')
    axarr[i].imshow(image_stream)
    axarr[i].set_title( L[i])
plt.show()
