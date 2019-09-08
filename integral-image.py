import numpy as np

image = np.random.random((5,5))
print(image)
integralImage = np.zeros((5,5))

for i in range(len(integralImage)):
    for j in range(len(integralImage[0])):
        left = 0
        top = 0  
        topLeft = 0
        if(j-1 >= 0):
            left = integralImage[i][j-1]
        if(i-1 >= 0):
            top = integralImage[i-1][j]
        if(i-1 >= 0 and j-1 >= 0):
            topLeft = integralImage[i-1][j-1]
        integralImage[i][j] = left + top - topLeft + image[i][j]
        
print(integralImage)