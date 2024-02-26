from cProfile import label
from calendar import c
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import math
import tkinter as tk
from tkinter import filedialog
import gc


added_files = []

def open_file(file_path):
    X_cord = []
    Y_cord = []
    Z_cord = []
    with open(file_path, 'r') as file:
        for idx, row in enumerate(file):
            column = row.split()
            if len(column) >= 3:
                X_cord.append(float(column[0]))
                Y_cord.append(float(column[1]))
                Z_cord.append(float(column[2]))
            else:
                print("bad format")

        # usuniecie bledu bazowania 
        max_Z = max(Z_cord)
        min_Z = min(Z_cord)
        adv = (max_Z + min_Z)/2
        Z_cord_n = [i - adv for i in Z_cord]
        Z_cord = Z_cord_n

        return X_cord, Y_cord, Z_cord

def create_step(X, Y):
    """
    local_steps = []
    for x, y in zip(X, Y):
        step = math.sqrt(x**2 + y**2)
        local_steps.append(step)
    step = sum(local_steps)/(len(local_steps)*100)
    
    local_steps = [] # zerowanie 
    for i in range(len(X)):
        local_steps.append(step*i)
    steps = local_steps
    """
    step = 1
    steps = []
    for i in range(len(X)):
        steps.append(i)
    return steps, step

def DFT_transform(x, z):
    z_np_array = np.array(z)
    # obliczenie dft 
    DFT_ = np.fft.fft(z_np_array)
    # oblicznie osi czestotliwosci
    len_DFT = len(DFT_)
    freq = np.fft.fftfreq(len_DFT)
    # amplituda 
    ampli = 2 * np.abs(DFT_) / len_DFT
    ampli_no_abs = 2* (DFT_)/len_DFT
    # przesuniece fazowe 
    phases = np.angle(DFT_)
    # filtracja dla ujemnych czestotliwosci 
    indeksy_nieujemne = np.where(freq >= 0)
    freq_nieujemne = freq[indeksy_nieujemne]
    ampli_nieujemne = ampli[indeksy_nieujemne]
    ampli_no_abs = ampli_no_abs[indeksy_nieujemne]
    phases_nieujemne = phases[indeksy_nieujemne]
    return freq_nieujemne, ampli_nieujemne, phases_nieujemne
    
def dominant_components(ampli, freq, phases, number, steps):
    # szukanie skladowych 
    sorted_ampli = sorted(ampli)
    top_ampli = sorted_ampli[-int(number):]
    indices = [i for i, val in enumerate(ampli) if val in top_ampli]
    top_ampli = ampli[indices]
    top_freq = freq[indices]
    top_phases = phases[indices]
    
    #lista cosinsow (jebac sinusy bo e^i*fi = cos(fi*z) + i sin(fi*z)  czyli sinus jest zespolony z cosinus rzeczywisty)
    formula = []
    t = [steps[1]*idx for idx in range(len(steps))]
    t = np.array(t)

    val_sum_graf = {}
    for i in range(len(top_ampli)):
        val_sum_graf[i] = []

    for idx, (i, j, k) in enumerate(zip(top_ampli, top_freq, top_phases)):
        if j == 0:
            val_sum_graf[idx] = [0]
            _formula = 0
            formula.append(_formula)
        else:
            _formula = (f"{round(i, 4)} * cos({round(2*3.14*j, 4)} * t + {round(k, 4)})")
            formula.append(_formula)
            val_sum_graf[idx].extend(i * np.cos(2 * np.pi * j * t + k))

    return val_sum_graf, formula, top_ampli, top_freq, top_phases

def plot_3d_graf(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    ax.set_xlabel('Oś X')
    ax.set_ylabel('Oś Y')
    ax.set_zlabel('Oś Z')
    plt.show()

def plot_2d(steps, z, freq, ampli):
    plt.subplot(2, 1, 1)
    plt.plot(steps, z)
    plt.title("dane z CMM")

    plt.subplot(2, 1, 2)
    plt.stem(freq, ampli)
    plt.xlim(-0.001, 0.01)
    plt.title("DFT")

    plt.legend()
    plt.show()


def plot_2d_v2(steps, z, dic, formulas): # bedzie trzeba wyjbac korelacje do osobnej funkcji 
    # wykres otrzymany
    values = [sum(val) for val in zip(*dic.values())]
    plt.subplot(3,1,1)
    plt.plot(steps, values, c='r', label = "suma interferencji")
    for idx, key in enumerate(dic):
       plt.scatter(steps, dic[key], s=1, label = f"{formulas[idx]}")
    plt.legend()

    # porownaie
    plt.subplot(3,1,2)
    plt.plot(steps, z, label = "CMM")
    plt.scatter(steps, values, s=4, c='r', label = "Suma")

    # oblicznie korelacji
    correlation = np.correlate(z, z, mode='full')
    time_corr = np.arange(-len(values) + 1, len(z))

    plt.subplot(3,1,3)
    plt.plot(time_corr, correlation)
    plt.title("Cross-correlation")
    plt.xlabel("Time Lag")

    plt.tight_layout()
    plt.legend()
    plt.show()

    correlation = np.corrcoef(z, values)[0,1]
    procentowe_pokrycie = (1 + correlation) / 2 * 100
    print(procentowe_pokrycie)

def laplace_transform(ampli, freq, k):
    L_o = []
    M_o = []
    for i, j in zip(ampli, freq):
        if j == 0:
            L_o.append([round(i, 6)])
            M_o.append([1, 0])
        else:
            L_o.append([1, 0])
            M_o.append([1, 0, round(2*np.pi*j, 6)**2])

    ampli_copy = ampli.tolist()
    freq_copy = freq.tolist()

    while True:
        input_index = np.argmin(freq)
        input_ampli = ampli[input_index]
        input_freq = freq[input_index]


        # Sprawdź, czy najmniejsza częstotliwość wynosi 0
        if input_freq == 0:
            # Jeśli tak, usuń element o najmniejszej częstotliwości
            np.delete(input_freq, input_index)
            np.delete(input_ampli, input_index)

            # Pobierz kolejny element o najmniejszej częstotliwości
            input_index = np.argmin(freq)
            input_ampli = ampli[input_index]
            input_freq = freq[input_index]
        else:
            break
    L_i = [input_ampli, 0]
    M_i = [1, 0, (2*3.14*input_freq)**2]

    tf_o_list = []
    for idx, (l, m) in enumerate(zip(L_o, M_o)):
        tf_o_list.append(ampli[idx]*ctrl.TransferFunction(l, m))

    for idx, tf in enumerate(tf_o_list):
        print(f"{idx + 1}: {tf}")

    # transformarmatqa sygnalu wyjsciowego
    tf_o = ctrl.parallel(*tf_o_list)
    # transformata sygnalu wejsiowego
    tf_i = ctrl.TransferFunction(L_i, M_i)

    G_open = tf_o/tf_i # ukjlad otwart

    # usuwanie stalej rozniczkownaia 
    num, den = ctrl.tfdata(G_open)

    # Sprawdź, czy ostatni element num i den jest bliski 0
    if np.any(np.isclose(np.real(num[-1]), 0)) and np.any(np.isclose(np.real(den[-1]), 0)):
        # Usuń ostatnie elementy, które są bliskie zeru
        num = num[0][0][:-1]
        den = den[0][0][:-1]

    G_open = ctrl.TransferFunction(num, den)
    K = ctrl.TransferFunction([k], [1])
    G_open = ctrl.series(G_open, K)
    feedback = ctrl.TransferFunction([1], [1])
    G_close_loop = ctrl.feedback(G_open, feedback)
    print (G_open)
    return G_open, G_close_loop, tf_i, tf_o, input_ampli, input_freq

def roots_of_transfer(G_open):

    num, den = ctrl.tfdata(G_open)
    zeros = ctrl.zeros(G_open)
    pole = ctrl.pole(G_open)
    k_num = [num[0][0][0]]
    k_den = [den[0][0][0]]

    # elemnt rzeczyw3isty jest pomijalnie maly dlatego uzywmay tylko czesci urojonej
    zeros = zeros.imag
    pole = pole.imag

    for zero, pol in zip(zeros, pole):
        k_num.append(zero)
        k_den.append(pol)

    # przypisane wartosci pocztakowej 1 zeby unikac bledu
    K_num = 1
    K_den = 1

    for k_nu, k_de in zip(k_num, k_den):
        K_num = K_num*k_nu
        K_den = K_den*k_de
    K = K_num/K_den
    print(K)
    return K

def plot_trnasfer(input_ampli, input_freq, G_open):
    # sprawdzenie czy program wreszcie prawidlowo okresla transmitancje 
    time = [idx for idx in range(5400)]
    time = np.array(time)

    omega = 2*np.pi*input_freq

    u = input_ampli * np.cos(omega*time)
    time, response = ctrl.forced_response(G_open, time, u)
    plt.plot(time, u, label = "sygnal wymuszajacy")
    plt.ylim(-0.15, 0.15 )
    plt.xlim(0, 5400)
    plt.scatter(time, response,s=1, c="r", label = "odpowiedz ukladu")
    plt.legend()


#### szuaknie min i max lokalnych 
def min_max(steps, z, top_ampli, top_freq, top_phases):
    # zamioana na np array
    top_ampli = np.array(top_ampli)
    top_freq = np.array(top_freq)
    top_phase = np.array(top_phases)

    # max ampli
    idx = np.argmax(top_ampli)
    top_ampli = top_ampli[idx]
    top_freq = top_freq[idx]
    top_phase = top_phase[idx]

    ######################################### pierwsza metoda ###################################
    # lcizenie oemgi
    omega = 2 * np.pi * top_freq
    peroid = 1/top_freq
    phase_shift = peroid - top_phase/omega
    peroid = int(peroid)
    phase_shift = int(phase_shift)

    #podzil na podlisty
    phase_Z = z[:phase_shift]
    sublist_z = []
    sublist_z.append(phase_Z)
    for i in range(len(phase_Z), len(z), peroid+30):
        sublist_z.append(z[i:i+peroid])

    max_list = []
    min_list = []
    for sublist in sublist_z:
        max_list.append(max(sublist))
        min_list.append(min(sublist))

    max_list_index = [z.index(max_val) for max_val in max_list]
    min_list_index = [z.index(min_val) for min_val in min_list]
  
    for i, j, k, m in zip(max_list, max_list_index, min_list, min_list_index):
        print(f"max: {i} indeksy: {j}")
        print(f"min: {k} indeksy: {m}")
    print(len(max_list), len(min_list))

    plt.plot(steps, z)
    plt.scatter(max_list_index, max_list, marker='x', color='red')
    plt.scatter(min_list_index, min_list, marker='x', color='blue')
    plt.show()



##################################################################################################################
##################################################################################################################


# tworzenie gui 

def calling_fun(file_paths):
    K_list = []
    for path in file_paths:
        gc.collect()
        x, y, z = open_file(path)
        steps, step = create_step(x, y)
        freq, ampli, phases = DFT_transform(x, z)
        dict_val, formulas, top_ampli, top_freq, top_phases = dominant_components(ampli, freq, phases, 2, steps)
        #plot_3d_graf(x, y, z)
        G_open, G_close, tf_i, tf_o, input_ampli, input_freq = laplace_transform(top_ampli, top_freq, 1)
        K = roots_of_transfer(G_open)
        K_list.append(K)
        print(K_list)


def file():
    filepaths = filedialog.askopenfilenames(
        initialdir="/",
        title="Wybierz Plik",
        filetypes=(("Pliki tekstowe", ".txt"), ("Wszystkie pliki", ".*"))
    )
    if filepaths:
        for filepath in filepaths:
            added_files.append(filepath)
            listbox.insert(tk.END, filepath)

def remove_file():
    listbox.delete(0, tk.END)
    added_files.clear()




root = tk.Tk()
root.title("dziwne akcje interpolacje")

frame_buttons = tk.Frame(root)
frame_buttons.pack(anchor="nw", padx=10, pady=10)

add_file_button = tk.Button(frame_buttons, text="Add File", command=file)
add_file_button.pack(side=tk.LEFT)

remove_file_button = tk.Button(frame_buttons, text="Remove Selected", command=remove_file)
remove_file_button.pack(side=tk.LEFT)

execiut_button = tk.Button(frame_buttons, text="Execute", command=lambda: calling_fun(list(added_files)))
execiut_button.pack(side=tk.RIGHT)

frame_listbox = tk.Frame(root)
frame_listbox.pack(anchor="nw", padx=10, pady=10)

listbox = tk.Listbox(frame_listbox)
listbox.pack()

root.mainloop()

