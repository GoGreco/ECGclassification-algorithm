#Importa as bibliotecas necessárias
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt

#Inicialização das variáveis
dataPath = '/content/physionet.org/files/mitdb/1.0.0'

selected_records = ['100','102','103','104']

duration_seconds = 5

fs = 360
n_samples = duration_seconds * fs

window_size = 15

recordPaths = []

#checa os arquivos ".hea" construindo o caminho para os arquivos selecionados
for rec in selected_records:
    full_path = os.path.join(dataPath, rec)
    if os.path.exists(full_path + '.hea'):
        recordPaths.append(full_path)
    else:
        print(f'Registro {rec} não encontrado.')

#Recebe um sinal e a janela a ser usada e retorna o sinal suavizado
def media_movel(sinal, janela):
    return np.convolve(sinal, np.ones(janela)/janela, mode='same')

#Processa o sinal e plota em uma gráfico, lê o sinal, corta na sessão de tempo especificada, aplica o filtro e mostra o gráfico
for recordPath in recordPaths:
    rec = wfdb.rdrecord(recordPath)
    sig = rec.p_signal
    n_channels = rec.n_sig
    names = rec.sig_name

    n_samples_clipped = min(n_samples, sig.shape[0])
    sig = sig[:n_samples_clipped, :]
    time = np.arange(n_samples_clipped) / fs

    sig_filtrado = np.zeros_like(sig)
    for i in range(n_channels):
        sig_filtrado[:, i] = media_movel(sig[:, i], window_size)

    fig, axes = plt.subplots(nrows=n_channels, ncols=1, figsize=(20, 3*n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, sig[:, i], label='Original', alpha=0.3)
        ax.plot(time, sig_filtrado[:, i], label='Filtrado (média móvel)', lw=1.0)
        ax.set_title(f'{os.path.basename(recordPath)} – {names[i]}')
        ax.set_ylabel('mV')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=15)
        ax.set_title(f'{os.path.basename(recordPath)} – {names[i]}', fontsize=20)
        ax.set_xlabel('Tempo (s)', fontsize=20)
        ax.set_ylabel('mV', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)

    axes[-1].set_xlabel('Tempo (s)')
    plt.tight_layout()
    plt.show()