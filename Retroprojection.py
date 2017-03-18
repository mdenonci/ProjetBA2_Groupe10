# -*- coding: utf-8 -*-
# ======================================================================
#
#       PROJET MULTIDISCIPLINAIRE - BA2 - 2016/17
#     Reconstruction Tomographique par Infrarouge --- Groupe 10
#   VERSION FINALE
#
#  ====== CODE PYTHON DE TRAITEMENT DES DONNEES ET RETROPROJECTION =====



# ==== Imports de Librairies ====

from PIL import Image as img
import numpy as np
import serial
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# === Definition des Constantes ===
'''
n_th := Nombre d'angles utilises
n_rh := Nombre de mesures sur UNE translation
Rotation_totale := 180 ou 360
'''

n_th = 72
n_rh = 950
Rotation_total = 360
att_coefficient = 0.55 # Coefficient d'attenuation
black_val = 933 # Valeur moyenne mesuree
margin = np.floor(n_rh/20)
artefact_tolerance = np.floor(n_rh/10)

theta = np.linspace(0, Rotation_total, n_th, endpoint=False)


# === Definition des Fonctions ===



def import_from_arduino(no_port):
    '''
     Cette fonction importe les donnees de l'arduino et retourne une liste de string
    '''
    ser = serial.Serial(port=no_port, baudrate=9600)
    donnees = []
    while (len(donnees)<= n_th*n_rh):
        donnees.append(ser.readline().strip().decode("utf-8"))
    return (donnees[1:len(donnees)+1])


    
def make_array(liste, colonnes, lignes):
    '''
    Cree un array (affichable en image) des dimensions souhaitees a partir d'une liste
    '''
    tab = np.array(liste)
    tab2 = tab.reshape(colonnes,lignes)
    final_tab = np.transpose(tab2)

    return final_tab



def measure_to_pixelvalue(measure_list, p_level):

    '''
    Transforme les donnees d'atenuation en niveau de gris
    '''
    L_list = []
    for measure in measure_list :
        L = np.abs(( 1/att_coefficient) * np.log(measure/black_val))
        L_list.append(L)
    print(L_list)
    greyscale_list = []
    color_inter = int(255 / p_level)
    max_measure = max(L_list)
    min_measure = min(L_list)
    delt_measure = max_measure - min_measure
    measure_inter = delt_measure / p_level

    liste_intervalle_gris = []
    for clr in range(0, p_level + 1):
        liste_intervalle_gris.append(clr * color_inter)

    color_numer = []
    for measure in L_list:
        inter_n = 1
        fint = measure_inter
        while measure - fint > 0:
            fint += measure_inter
            inter_n += 1
        else:
            if inter_n > len(liste_intervalle_gris):
                inter_n = len(liste_intervalle_gris) - 1

            color_numer.append(inter_n)

    for color in color_numer:
        if color == 0:
            greyscale_list.append(0)
        else:
            greyscale_list.append(liste_intervalle_gris[color - 1])
    print(greyscale_list)
    return greyscale_list


    
def iradon(sinogram,filter_used):
    '''
    Fonction de Retroprojection
    '''
    # Definition des Valeurs Necessaires
    th = (np.pi / 180.0) * theta                                                            
    output_size = n_rh
    f = fftfreq(n_rh).reshape(-1,1)                                                         
    omega = 2 * np.pi * f
    ramp_filter = np.abs(f)  # definition du filtre rampe

    # Definition du filtre selon le filtre choisi en parametre
    if filter_used is None:
        filter_def = 0.01 # 0.1 et non 1 afin de rendre la reconstruction visible
    elif filter_used == "Rampe":
        filter_def = ramp_filter
    elif filter_used == "Hamming":
        filter_def = ramp_filter * (0.54 + 0.46 * np.cos(omega / 2)) #Definition du filtre Hamming
    elif filter_used == "Shepp-Logan":
        ramp_filter[1:] = ramp_filter[1:] * np.sin(omega[1:]) / omega[1:]
        filter_def = ramp_filter
    else:
        print("Le filtre entre est incorrect, par defaut le filtre rampe est applique")
        filter_def = ramp_filter

    # Application du filtre dans l'espace de Fourier et retour dans l'espace 'reel'
    filtered_FT = fft(sinogram, axis=0) * filter_def # Application du filtre
    new_proj = (np.real(ifft(filtered_FT, axis = 0))) # Retour dans l'espace reel

    # Definition des Parametres de Reconstruction
    center = sinogram.shape[0] // 2
    reconstructed = np.zeros((output_size,output_size))
    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2
    for i in range(len(th)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
        x = np.arange(new_proj.shape[0]) - center
        backprojected = np.interp(t, x, new_proj[:, i],left=0, right=0)
        reconstructed += backprojected
        
    return reconstructed



def correct_artefact(limit_pixel_list, tolerance):
    for b in range(len(limit_pixel_list)-1):
        if b <= 3:
            if np.abs(limit_pixel_list[b] - limit_pixel_list[b+3]) > tolerance:
                limit_pixel_list[b] = limit_pixel_list[b+3]
        elif b >= len(limit_pixel_list)-3 :
            if np.abs(limit_pixel_list[b] - limit_pixel_list[b-3]) > tolerance:
                limit_pixel_list[b] = limit_pixel_list[b - 3]
        else:
            if np.abs(limit_pixel_list[b] - limit_pixel_list[b+3]) > tolerance and \
                            np.abs(limit_pixel_list[b] - limit_pixel_list[b-3]) > tolerance:
                limit_pixel_list[b] = limit_pixel_list[b+3]
    return limit_pixel_list

def find_up_limit(sinogram_array):
    u_pixel_transition = []
    for column in range(n_th):
        line = 0
        pixel_val = sinogram_array[line +1, column]
        while (np.abs((pixel_val - sinogram_array[line, column]) < 5)) and line+1 != n_rh:
            line += 1
            pixel_val = sinogram_array[line + 1, column]
        u_pixel_transition.append(line)
    return u_pixel_transition


def find_down_limit(sinogram_array):
    d_pixel_transition = []
    for column_b in range(n_th):
        line = n_rh - 1
        pixel_val = sinogram_array[line - 1, column_b]
        while (pixel_val - sinogram_array[line, column_b] < 5):
            line -= 1
            pixel_val = sinogram_array[line - 1, column_b]
        d_pixel_transition.append(line)
    return d_pixel_transition

def Recentering(sinogram_array,margin,minimal_up,maximal_down):
    print("minup : ", minimal_up, " maxdwn : ", maximal_down)
    while (sinogram_array.shape)[0] != (n_rh - minimal_up + margin):
        sinogram_array = np.delete(sinogram_array, 0, 0)

    maximal_down_inverted = n_rh - maximal_down
    while (sinogram_array.shape)[0] != (n_rh - minimal_up - maximal_down_inverted + (2*margin)):
        sinogram_array = np.delete(sinogram_array, (sinogram_array.shape)[0] - 1, 0)
    return sinogram_array

def correct_sinogram(sinogram_array):
    for a in range(n_th):
        tabtr = sinogram_array.transpose()
        if a % 2 == 0:
            to_reverse = tabtr[a]
            tabtr[a] = to_reverse[::-1]

    return tabtr.transpose()

# === SCRIPT ===

# ETAPE 1 : Import des Donnees de l'Arduino et conversion des STR en FLOAT
mesures = import_from_arduino('COM5')

print(mesures)
print(len(mesures))

print("IMPORTATION DES MESURES TERMINEE")
for i in range(0,len(mesures)):
    mesures[i] = float(mesures[i])

measure_tab = make_array(mesures,n_th,n_rh)
np.savetxt('Mesures4.txt',measure_tab)


# ETAPE 2 : Conversion des donnees brutes en donnees utiles
liste_color = measure_to_pixelvalue(mesures,255)

print("CONVERSION DES DONNEES TERMINEE")

# ETAPE 3 : Creation d'un Array de dimensions souhaitees = Array de Sinogramme  , Conversion en image et affichage
tab_color = make_array(liste_color, n_th, n_rh)

tab_color = correct_sinogram(tab_color)

plt.plot(np.arange(n_rh),(tab_color[0:n_rh]))
plt.show()

sinogram_img = img.fromarray(tab_color)
print("SINOGRAMME CREE")
sinogram_img.show()
sinogram_img.convert('RGB').save('Path/file.filetype')


# ETAPE 4 : Recentrage du Sinogramme

up_pixel_transition = find_up_limit(tab_color)
down_pixel_transition = find_down_limit(tab_color)


up_pixel_transition = correct_artefact(up_pixel_transition, artefact_tolerance)
down_pixel_transition = correct_artefact(down_pixel_transition, artefact_tolerance)

print(up_pixel_transition)
print(down_pixel_transition)

minimal_up = up_pixel_transition[0]
for upix in up_pixel_transition:
    if upix < minimal_up:
        minimal_up = upix

maximal_down = down_pixel_transition[0]
for upix in down_pixel_transition:
    if upix > maximal_down:
        maximal_down = upix


tab_color_recentered = Recentering(tab_color, margin,minimal_up,maximal_down)

centered_sinogram_img = img.fromarray(tab_color_recentered)
print("SINOGRAMME RECENTRE")
centered_sinogram_img.show()
centered_sinogram_img.convert('RGB').save('Path/file.filetype')

n_rh = (tab_color_recentered.shape)[0]

# ETAPE 5 : Reconstruction, conversion en image, affichage final et sauvegarde
reconstruction = iradon(tab_color_recentered, "Shepp-Logan")

reconstruction_img = img.fromarray(reconstruction)
print("RECONSTRUCTION TERMINEE")
reconstruction_img.show()
reconstruction_img.convert('RGB').save('Path/file.filetype')
            
# =================================================================================================
