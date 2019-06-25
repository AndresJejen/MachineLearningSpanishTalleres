import numpy as np

def sigmoid(Z):
    """
    Implementa la activacion sigmoide con numpy
    Input:
    Z: arreglo numpy
    Output:
    A: salida de sigmoid(z), mismo tamano que Z
    cache: devuelve Z
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implementa la activacion RELU.
    Input:
    Z: salida de la capa lineal 
    Output:
    A: parametro de post-activation, mismo tamano que Z
    cache: diccionario python con "A" 
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implementa la retro-propagacion para una sola unidad RELU.
    Input:
    dA: gradiente de post-activacion
    cache: se guarda 'Z' para la retro-propagacion 
    Output:
    dZ: Gradiente del coste con respecto a Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # se convierte dz al objeto correcto
    
    dZ[Z <= 0] = 0    # Cuando z <= 0, dz tambien es 0. 
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implementa la retro-propagacion para una sola unidad SIGMOIDE.
    Input:
    dA: gradiente de post-activacion
    cache: se guarda 'Z' para la retro-propagacion 
    Output:
    dZ: Gradiente del coste con respecto a Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
