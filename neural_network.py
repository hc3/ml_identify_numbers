from time import time  # contabilizar o treino

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # para utilizar a função de ativação, etc.
import torchvision  # datasets populares, modelos de arquitetura, e transf. de imagens
from torch import nn, optim  # optim implementa vários algoritmos de otimização
from torchvision import datasets, transforms # transformar de imagens para tensores, etc.
from PIL import Image
import matplotlib.pyplot as plt
from invert import Invert

ENDERECO_DO_MODELO_TREINADO = './db/model_weights.pth'

transform = transforms.Compose([Invert(),
                                    transforms.ToTensor(), transforms.Resize(size = (28,28) )])
# transform = transforms.ToTensor() #definindo a conversão de imagem para tensor

trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform) # Carrega a parte de treino do dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) # Cria um buffer para pegar os dados por partes

valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform) # Carrega a parte de validação do dataset
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True) # Cria um buffer para pegar os dados por partes

class Modelo(nn.Module):
    def __init__(self): #aqui em init definimos a arquitetura da rede
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28, 128) # camada de entrada, 784 neurônios que se ligam a 128
        self.linear2 = nn.Linear(128, 64) # camada interna 1, 128 neurônios que se ligam a 64
        self.linear3 = nn.Linear(64, 10) # camada interna 2, 64 neurônios que se ligam a 10
        # para a camada de saida não e necessário definir nada pois só precisamos pegar o output da camada interna 2
        
    def forward(self,X): # aqui a partir do X obtemos o Y
        X = F.relu(self.linear1(X)) # função de ativação da camada de entrada para a camada interna 1
        X = F.relu(self.linear2(X)) # função de ativação da camada interna 1 para a camada interna 2
        X = self.linear3(X) # função de ativação da camada interna 2 para a camada de saída, nesse caso f(x) = x
        return F.log_softmax(X, dim=1) # dados utilizados para calcular a perda


def treino(modelo, trainloader, device):
    #SGD = stochastic gradient descent, lr = learning rate
    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5) # define a política de atualização dos pesos e da bias
    inicio = time() # timer para sabermos quanto tempo levou o treino
    
    criterio = nn.NLLLoss() # definindo o criterio para calcular a perda
    EPOCHS = 30 # numero de epochs que o algoritmo rodará
    modelo.train() # ativando o modo de treinamento do modelo

    for epoch in range(EPOCHS):
        perda_acumulada = 0 # inicialização da perda acumulada da epoch em questão
        
        for imagens, etiquetas in trainloader:
            
            imagens = imagens.view(imagens.shape[0], -1) # convertendo as imagens para  "vetores" de 28*28 casas para ficarem compatíveis com a camada de entrada
            otimizador.zero_grad() # zerando os gradientes por conta do ciclo anterior
            
            output = modelo(imagens.to(device)) # colocando os dados no modelo
            perda_instantanea = criterio(output, etiquetas.to(device)) # calculando a perda da epoch em questão
            
            perda_instantanea.backward() # back propagation a partir da perda
            
            otimizador.step() # executa a otimização atualizando os pesos e a bias
            
            perda_acumulada += perda_instantanea.item() # atualização da perda acumulada
        
        
        else:
            print("Epoch {} - Perda resultante: {}".format(epoch+1, perda_acumulada/len(trainloader)))
    print("\nTempo de treino (em minutos) =",(time()-inicio)/60)
    torch.save(modelo, ENDERECO_DO_MODELO_TREINADO)

def coloca_fundo_branco(im):
    data = np.array(im)

    r1, g1, b1 = 0, 0, 0 # Original value
    r2, g2, b2 = 255, 255, 255 # Value that we want to replace it with

    red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:,:,:3][mask] = [r2, g2, b2]

    im = Image.fromarray(data)
    return im
    # im.save('fig1_modified.png')

def validacao(modelo, valloader, device):
    conta_corretas, conta_todas = 0, 0
    for imagens,etiquetas in valloader:
      for i in range(len(etiquetas)):
        img = imagens[i].view(1, 784) # view muda o formato, the shape
        # desativar o autograd para acelerar a validação. Grafos computacionais dinâmicos tem um custo alto de processamento
        with torch.no_grad():
            logps = modelo(img.to(device)) # saída do modelo em escala logaritmica

        
        ps = torch.exp(logps) # converte a saída para escala normal(lembrando que é um tensor)
        probab = list(ps.cpu().numpy()[0]) #cpu move o tensor para o cpu
        etiqueta_pred = probab.index(max(probab)) # converte o tensor em um número, no caso, o número que o modelo previu como correto 
        etiqueta_certa = etiquetas.numpy()[i] 
        if(etiqueta_certa == etiqueta_pred): # compara a previsão com o valor correto
          conta_corretas += 1
        conta_todas += 1

    print("Total de imagens testadas =", conta_todas)
    print("\nPrecisão do modelo = {}%".format(conta_corretas*100/conta_todas))
    return conta_corretas*100/conta_todas

def visualiza_pred(img, ps):
    # x = coloca_fundo_branco(img)
    ps = ps.data.cpu().numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.numpy().squeeze(), cmap='gray_r')
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Palpite')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


modelo = torch.load(ENDERECO_DO_MODELO_TREINADO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # modelo rodará na GPU se possível
modelo.to(device)

if not modelo:
    treino(modelo, trainloader, device)

#setGrayScale = transforms.Grayscale()
#resize_function = transforms.Resize(size = (28,28))

img = Image.open('./my_samples/numero_2_camera_celular.jpeg')
#img = setGrayScale(img)
#img = resize_function(img)
#img_resize.show()

image_em_formato_de_tensor = transform(img)[0].view(1, 784)

# image_em_formato_de_tensor = image_em_formato_de_tensor.where(image_em_formato_de_tensor >= 1, image_em_formato_de_tensor * 10)

with torch.no_grad():
    logps = modelo(image_em_formato_de_tensor.to(device))

ps = torch.exp(logps)
probab = list(ps.cpu().numpy()[0])

# print(coloca_fundo_branco(img_resize))

print("Número previsto =", probab.index(max(probab)))
#print('img resize: ',img_resize)
#print('images : ',asdasdas)
#plt.imshow(imgkk.view(1, 28, 28))
#plt.show()
#visualiza_pred(coloca_fundo_branco(resize_function), ps)
visualiza_pred(image_em_formato_de_tensor.view(1, 28, 28), ps)
# print(image_em_formato_de_tensor)
# plt.imshow(coloca_fundo_branco(img_resize))
# plt.imshow(img)
# plt.show()
# visualiza_pred(image_em_formato_de_tensor.view(1, 28, 28), ps)

# print(image_em_formato_de_tensor)
# print(img_resize)
# plt.imshow(image_em_formato_de_tensor.view(1, 28, 28))
# plt.show()
# print(image_em_formato_de_tensor)
# print(next(iter(valloader))[0][0])
# validacao(modelo, valloader, device)

# imagens, etiquetas = next(iter(valloader))

# img = imagens[0].view(1, 784)
# plt.imshow(imagens[0].view(1, 28, 28))
# plt.show()
# # print(imagens[0])
# with torch.no_grad():
#     logps = modelo(img.to(device))

# ps = torch.exp(logps)
# probab = list(ps.cpu().numpy()[0])
# print("Número previsto =", probab.index(max(probab)))

# # def testando():
# #     l
# # visualiza_pred(img.view(1, 28, 28), ps)