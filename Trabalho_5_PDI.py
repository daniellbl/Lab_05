import cv2
import numpy as np
from matplotlib import pyplot as plt

#---------- Limiarização --------------
# Lê a imagem
img0 = cv2.imread('lenat.bmp') 

# converte para escala de cinza
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove o ruido
img = cv2.GaussianBlur(gray,(3,3),0)

# Aplica o filtro Laplaciano
laplacian = cv2.Laplacian(img,cv2.CV_64F)

# Aplica a limiarização com T = 20
ret,thresh1 = cv2.threshold(laplacian,20,255,cv2.THRESH_BINARY)


# Junta as imagens para exibição lado a lado
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(thresh1,cmap = 'gray')
plt.title('Limiarização'), plt.xticks([]), plt.yticks([])

#Exibe as imagens
plt.show()

#-------Transformada de Hough para círculos -----------------------------
#Lê a imagem
img = cv2.imread('circle.png')
# converte para escala de cinza
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Aplica um filtro de média para reduzir ruído
img = cv2.medianBlur(img,5)
# Converte a imagem para RGB
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# Aplica a transformada de Hough para círculos
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=100,minRadius=60,maxRadius=0)
# Converte o tipo da variavel
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # desenha o circulo
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # desenha o centro do circulo
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

# Junta as imagens para exibição lado a lado
plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(cimg)
plt.title('Hough'), plt.xticks([]), plt.yticks([])

#Exibe as imagens
plt.show()

#-----------Transformada de Hough para linhas -----------------------------
#Lê a imagem
img = cv2.imread('lines.jpg')
#Converte para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Pode se usar Canny ou o Limiar para HoughLines()
#ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Executa a transformação de Hough para linhas.  
lines = cv2.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            # Desenha a linha
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# Lê a imagem original
img_original = cv2.imread('lines.jpg')

# Junta as imagens para exibição lado a lado
plt.subplot(1,2,1),plt.imshow(img_original)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(img)
plt.title('Hough'), plt.xticks([]), plt.yticks([])

#Exibe as imagens
plt.show()

#--------K-Means para imagens coloridas---------------------

# Lê a imagem
image = cv2.imread('ave.png')

# Converte para RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Modela a imagem para um vetor 2D de pixels e 3 canais de cores (RGB)
pixel_values = image.reshape((-1, 3))

# Converte para float
pixel_values = np.float32(pixel_values)

# Define uma condição de parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Número de clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Converte de volta para 8 bit 
centers = np.uint8(centers)

# Compacta a matriz de etiquetas
labels = labels.flatten()

# Converte todos os pixels para a cor dos centroids
segmented_image = centers[labels.flatten()]

# Modela de volta para a dimensão da imagem original
segmented_image = segmented_image.reshape(image.shape)


# Junta as imagens para exibição lado a lado
plt.subplot(1,2,1),plt.imshow(image)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(segmented_image)
plt.title('K-Means'), plt.xticks([]), plt.yticks([])

#Exibe as imagens
plt.show()

#--------K-Means para imagens em tons de cinza ---------------------

# Lê a imagem
image = cv2.imread('carro.png')

#Converte para escala de cinza
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Modela a imagem para um vetor 2D de pixels e 1 canal
pixel_values = image.reshape((-1, 1))

# Converte para float
pixel_values = np.float32(pixel_values)

# Define uma condição de parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Número de clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Converte de volta para 8 bit 
centers = np.uint8(centers)

# Compacta a matriz de etiquetas
labels = labels.flatten()

# Converte todos os pixels para a cor dos centroids
segmented_image = centers[labels.flatten()]

# Modela de volta para a dimensão da imagem original
segmented_image = segmented_image.reshape(image.shape)

# Junta as imagens para exibição lado a lado
plt.subplot(1,2,1),plt.imshow(image,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(segmented_image,cmap = 'gray')
plt.title('K-Means'), plt.xticks([]), plt.yticks([])

#Exibe as imagens
plt.show()




























