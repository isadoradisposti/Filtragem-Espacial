import cv2
import numpy as np
import matplotlib.pyplot as plt

#Função de Convolução usando o OpenCV
def convolucao_opencv(imagem, kernel):
    return cv2.filter2D(imagem, -1, kernel)

media = np.ones((3, 3)) / 9
gaussiano = cv2.getGaussianKernel(5, 1) * cv2.getGaussianKernel(5, 1).T
laplaciano = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

#Carregando as imagens
imagens_info = {
    "lena": {
        "path": "lena_gray_512.tif",
        "data": None
    },
    "biel": {
        "path": "biel.png",
        "data": None
    },
    "cameraman": {
        "path": "cameraman.tif",
        "data": None
    }
}

diretorio = "./imagem/"
for nome, info in imagens_info.items():
    imagem = cv2.imread(diretorio + info["path"], cv2.IMREAD_GRAYSCALE)
    imagens_info[nome]["data"] = imagem

for nome, info in imagens_info.items():
    imagem = info["data"]

    imagem_media = convolucao_opencv(imagem, media)
    imagem_gauss = convolucao_opencv(imagem, gaussiano)
    imagem_laplac = convolucao_opencv(imagem, laplaciano)
    imagem_sobel_x = convolucao_opencv(imagem, sobel_x)
    imagem_sobel_y = convolucao_opencv(imagem, sobel_y)
    imagem_gradiente = np.sqrt(imagem_sobel_x*2 + imagem_sobel_y*2)
    imagem_laplac_original = imagem + imagem_laplac

    fig, axs = plt.subplots(1, 8, figsize=(25, 5))
    axs[0].imshow(imagem, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(imagem_media, cmap='gray')
    axs[1].set_title('Média')
    axs[1].axis('off')
    axs[2].imshow(imagem_gauss, cmap='gray')
    axs[2].set_title('Gaussiano')
    axs[2].axis('off')
    axs[3].imshow(imagem_laplac, cmap='gray')
    axs[3].set_title('Laplaciano')
    axs[3].axis('off')
    axs[4].imshow(imagem_sobel_x, cmap='gray')
    axs[4].set_title('Sobel X')
    axs[4].axis('off')
    axs[5].imshow(imagem_sobel_y, cmap='gray')
    axs[5].set_title('Sobel Y')
    axs[5].axis('off')
    axs[6].imshow(imagem_gradiente, cmap='gray')
    axs[6].set_title('Gradiente')
    axs[6].axis('off')
    axs[7].imshow(imagem_laplac_original, cmap='gray')
    axs[7].set_title('Laplac + Original')
    axs[7].axis('off')

    plt.tight_layout()
    plt.show()