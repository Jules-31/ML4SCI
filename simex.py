import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import time
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# RBM
class RBMGenerador(nn.Module):
    '''
    Red neuronal probabilística la cual aprende la distribución de probabilidad
    de sus entradas, por lo que se puede generar datos nuevos con la misma distribución
    en caso de tenerlos.
    En la capa visible (v) entran los datos y en la oculta (h) se aprenden los patrones. 
    La probabilidad conjunta de ambas capas está dada por la distribución de Boltzmann
    la cual describe la probabilidad P(v,h) que una partícula ocupe un estado de energía
    E(v,h) a una temperatura T. 
    - E(v,h) = -v^T*w*h - b_v*v - b_h*h
    - P(v,h) = exp(-E(v,h)) / Z
    Con:
        - v, el vectore de estado de las neuronas visibles.
        - b_v, el vectore de sesgos de las neuronas visibles.
        - h, el vectore de estado de las neuronas ocultas.
        - b_h, el vectore de sesgos de las neuronas ocultas.
        - w, la matriz de pesos por las conexiones entre v y h.  
        - Z, la función de partición la cual es la suma de todas las
             configuraciones posibles entre v y h.
    Para el aprendizaje, un vector v activa una neurona oculta donde se realiza una
    reconstrucción de la entrada incial. La diferencia entre ambas permite evaluar un error
    para ajustar los pesos por lo que se utiliza:
    - grad = <v·h>_data - <v·h>_model
    '''
    
    def __init__(self, n_visible, n_hidden, temperatura=1.0):
        '''
        n_visible: Número de neuronas en la capa visible.
        n_hidden: Número de neuronas en la capa oculta.
        temperatura: Controla la exploración, si es mayor que 1 es maás exploratoria.
        '''
        super(RBMGenerador, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.device = device
        self.temperatura = temperatura
        
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.to(device)
    
    def forward(self, v):
        '''
        Probabilidad de activación de las neuronas ocultas.
        '''
        h_prob = torch.sigmoid((torch.mm(v, self.W) + self.h_bias) / self.temperatura)
        return h_prob
    
    def sample_hidden(self, v):
        '''
        Calcular probabilidad de las neuronas y hace un muestro aleatorio en base a esta.
        '''
        h_prob = self.forward(v)
        h_sample = torch.bernoulli(h_prob)
        return h_prob, h_sample
    
    def sample_visible(self, h):
        '''
        Reconstrucción de las neuronal visibles a partir de las ocultas. 
        '''
        v_prob = torch.sigmoid((torch.mm(h, self.W.t()) + self.v_bias) / self.temperatura)
        v_sample = torch.bernoulli(v_prob)
        return v_prob, v_sample
    
    def gibbs_step(self, v):
        '''
        Ciclo de muestreo
        '''
        h_prob, h_sample = self.sample_hidden(v)
        v_prob, v_sample = self.sample_visible(h_sample)
        return v_prob, v_sample, h_prob
    
    def contrastive_divergence(self, v, k=1, lr=0.01):
        '''
        Aprendizaje y actualización de pesos
        '''
        h_prob_pos, h_sample_pos = self.sample_hidden(v)
        positive_grad = torch.mm(v.t(), h_prob_pos)
        
        v_neg = v.clone()
        for _ in range(k):
            _, v_neg, _ = self.gibbs_step(v_neg)
        
        h_prob_neg, _ = self.sample_hidden(v_neg)
        negative_grad = torch.mm(v_neg.t(), h_prob_neg)
        
        grad_W = (positive_grad - negative_grad) / v.size(0)
        grad_vb = torch.mean(v - v_neg, dim=0)
        grad_hb = torch.mean(h_prob_pos - h_prob_neg, dim=0)
        
        self.W.data += lr * grad_W
        self.v_bias.data += lr * grad_vb
        self.h_bias.data += lr * grad_hb
        
        return torch.mean((v - v_neg) ** 2).item()
    
    def entrenar(self, datos, epochs=80, batch_size=32, lr=0.01, k=1, nombre="RBM"):
        '''
        Bucle de entrenamiento
        '''
        n_samples = datos.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        if not torch.is_tensor(datos):
            datos = torch.tensor(datos, dtype=torch.float32).to(self.device)
        
        losses = []
        
        print(f"\n{'='*70}")
        print(f"ENTRENANDO RBM: {nombre}")
        print(f"Muestras: {n_samples} | Visibles: {self.n_visible} | Ocultos: {self.n_hidden}")
        print(f"{'='*70}\n")
        
        inicio = time.time()
        
        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            datos_shuffled = datos[indices]
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch = datos_shuffled[start_idx:end_idx]
                loss = self.contrastive_divergence(batch, k=k, lr=lr) # Aprendizaje
                epoch_loss += loss
            
            epoch_loss /= n_batches
            losses.append(epoch_loss)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                elapsed = time.time() - inicio
                if elapsed < 60:
                    tiempo = f"{elapsed:.1f}s"
                else:
                    tiempo = f"{elapsed/60:.1f}m"
                print(f"Epoca {epoch+1:4d}/{epochs} | Loss: {epoch_loss:.6f} | Tiempo: {tiempo}")
        
        print(f"\nRBM entrenada en {tiempo}")
        return losses
    
    def generar_muestras(self, n_samples, n_steps=100, return_probs=True, semilla=None):
        '''
        Generación de muevos datos a partir del entrenamiento de la RBM.
        '''
        if semilla is not None:
            torch.manual_seed(semilla)
            np.random.seed(semilla)
        
        self.eval()
        with torch.no_grad():
            v = torch.bernoulli(torch.ones(n_samples, self.n_visible) * 0.5).to(self.device)
        
            for step in range(n_steps):
                _, v, _ = self.gibbs_step(v)
                if step < n_steps * 0.3:
                    noise = torch.randn_like(v) * 0.05 * (1 - step / (n_steps * 0.3))
                    v = torch.clamp(v + noise, 0, 1)
            
            if return_probs:
                return v.cpu().numpy()
            else:
                return torch.bernoulli(v).cpu().numpy()
    
    def extraer_caracteristicas(self, v):
        '''
        Extrae características.
        '''
        with torch.no_grad():
            h_prob = self.forward(v)
        return h_prob
    
    def reconstruir(self, v, n_steps=10):
        '''
        Reconstruye datos de entrada
        '''
        v_curr = v.clone()
        for _ in range(n_steps):
            _, v_curr, _ = self.gibbs_step(v_curr)
        return v_curr
    
    def guardar_modelo(self, ruta="rbm_entrenada.pth"):
        '''
        Guardar los pesos de la RBM en un archivo.
        '''
        torch.save({
            'n_visible': self.n_visible,
            'n_hidden': self.n_hidden,
            'temperatura': self.temperatura,
            'device': self.device,
            'W': self.W.data,
            'v_bias': self.v_bias.data,
            'h_bias': self.h_bias.data
        }, ruta)
        print(f"  RBM guardada en: {ruta}")
    
    @classmethod
    def cargar_modelo(cls, ruta="rbm_entrenada.pth"):
        '''
        Cargar una RBM desde un archivo guardado.
        Cargar en CPU primero para seguridad
        '''
        checkpoint = torch.load(ruta, map_location=torch.device('cpu'))
        
        rbm = cls(
            n_visible=checkpoint['n_visible'],
            n_hidden=checkpoint['n_hidden'],
            temperatura=checkpoint['temperatura']
        )

        rbm.W.data = checkpoint['W']
        rbm.v_bias.data = checkpoint['v_bias']
        rbm.h_bias.data = checkpoint['h_bias']
        
        rbm.to(rbm.device)
        
        print(f"  RBM cargada desde: {ruta}")
        return rbm
    
# Visualizar entrenamiento
class EntrenamientoVisualizador:
    '''
    Visualizador en tiempo real del entrenamineto de la red neuronal, por medio de una barra
    de progreso se puede ver el tiempo estimado de entrenamiento.
    Las gráficas de evolución de pérdida y precisión se actualizan en tiempo real.
    '''
    def __init__(self, total_epochs, nombre_modelo="Modelo", guardar_graficas=True, directorio="./graficas/"):
        '''
        Inicialización de los parámetros de entrenamineto.
        total_epochs (int): Número total de épocas de entrenamiento.
        nombre_modelo (str): Nombre del modelo.
        '''
        self.total_epochs = total_epochs
        self.nombre_modelo = nombre_modelo
        self.guardar_graficas = guardar_graficas
        self.directorio = directorio
        self.inicio = None
        self.historial = {
            'epoch': [], 
            'train_loss': [], 
            'test_loss': [], 
            'train_r2': [], 
            'test_r2': [], 
            'time': []
        }
        self.mejor_loss = float('inf')
        self.fig = None
        self.axes = None
        if self.guardar_graficas:
            if not os.path.exists(self.directorio):
                os.makedirs(self.directorio)
                print(f"  Directorio creado para las gráficas: {self.directorio}")
    
    def _obtener_nombre_archivo(self, extension="png"):
        """
        Nombre archivo de las gráficas
        """
        from datetime import datetime
        
        ahora = datetime.now()
        fecha_str = ahora.strftime("%d%m%Y")
        hora_str = ahora.strftime("%H%M%S")
        
        # Quitar caracteres especiales
        nombre_limpio = self.nombre_modelo.replace(" ", "_").replace("/", "_").replace("\\", "_")
        nombre_limpio = ''.join(c for c in nombre_limpio if c.isalnum() or c in ['_', '-'])

        nombre_archivo = f"{nombre_limpio}_{fecha_str}_{hora_str}.{extension}"
        
        return nombre_archivo
    
    def _guardar_grafica(self):
        """
        Guarda la gráfica.
        """
        if self.fig is None or not self.guardar_graficas:
            return
        
        try:
            nombre_archivo = self._obtener_nombre_archivo("png")
            ruta_completa = os.path.join(self.directorio, nombre_archivo)
            
            self.fig.savefig(
                ruta_completa,
                dpi=300,              
                bbox_inches='tight',      
                facecolor='white',         
                edgecolor='none'
            )
            
            print(f"\n  Grafica guardada: {ruta_completa}")
            
        except Exception as e:
            print(f"\n  Error al guardar la grafica: {e}")
            
    def iniciar(self):
        '''
        Inicia visualización
        '''
        self.inicio = time.time()
        print(f"Entrenamiento: {self.nombre_modelo}")
        print(f"Total de epocas: {self.total_epochs}")
        print(f"Dispositivo: {device}")
        
        plt.ion() # Actualización en tiempo real de la gráfica
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 5))
        self.fig.suptitle(f'{self.nombre_modelo} - Evolucion del Entrenamiento', fontsize=14)
    
    def actualizar(self, epoch, train_loss, test_loss, train_r2=None, test_r2=None, mejorado=False):
        '''
        Actualiza el progreso de entrenamiento con las épocas.
        epoch: Número de época actuales.
        train_loss: Pérdida en el conjunto de entrenamiento.
        test_loss: Pérdida en el conjunto de validación.
        train_r2: R² en entrenamiento.
        test_r2: R² en validación.
        mejorado: True si el test_loss mejoró respecto a la época anter
        '''
        elapsed = time.time() - self.inicio
        
        self.historial['epoch'].append(epoch + 1) #Para mostrar desde 1
        self.historial['train_loss'].append(train_loss)
        self.historial['test_loss'].append(test_loss)
        self.historial['time'].append(elapsed)
        if train_r2 is not None:
            self.historial['train_r2'].append(train_r2)
        if test_r2 is not None:
            self.historial['test_r2'].append(test_r2)
        
        if epoch > 0:
            tiempo_por_epoch = elapsed / (epoch + 1)
            tiempo_restante = tiempo_por_epoch * (self.total_epochs - (epoch + 1))
            eta = datetime.now() + timedelta(seconds=tiempo_restante)
            eta_str = eta.strftime('%H:%M:%S')
        else:
            eta_str = "Calculando"
        
        if elapsed < 60:
            tiempo_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            tiempo_str = f"{elapsed/60:.1f}m"
        else:
            tiempo_str = f"{elapsed/3600:.1f}h"
        
        progreso = (epoch + 1) / self.total_epochs * 100
        barra = '█' * int(progreso / 2) + '░' * (50 - int(progreso / 2))
        estado = "Mejora" if mejorado else "  "
        
        r2_str = f" R2 Test: {test_r2:.4f}" if test_r2 is not None else ""
        
        msg = (f"\rEpoca {epoch+1:4d}/{self.total_epochs} "
               f"[{barra}] {progreso:5.1f}% | "
               f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}"
               f"{r2_str} | "
               f"Tiempo: {tiempo_str} | ETA: {eta_str} {estado}")
        
        if epoch == self.total_epochs - 1 or (epoch + 1) % 10 == 0:
            print(msg)
        else:
            print(msg, end='', flush=True)
        
        if epoch % 5 == 0 or epoch == self.total_epochs - 1:
            self._actualizar_graficas()
        
        if test_loss < self.mejor_loss:
            self.mejor_loss = test_loss
            return True
        return False
    
    def _actualizar_graficas(self):
        '''
        Actualización de las gráficas con lo datos guardados.
        '''
        if self.fig is None or self.axes is None:
            return
        
        for ax in self.axes:
            ax.clear()
        
        # Evolución pérdida
        ax = self.axes[0]
        if len(self.historial['train_loss']) > 0:
            ax.plot(self.historial['epoch'], self.historial['train_loss'], label='Train Loss', color="#8d36c7", linestyle='--', linewidth=2)
            ax.plot(self.historial['epoch'], self.historial['test_loss'], label='Test Loss', color="#df7cc9", linewidth=2)
            ax.set_xlabel('Epoca')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Evolucion de la Perdida')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # Evolución precisión
        ax = self.axes[1]
        if len(self.historial.get('train_r2', [])) > 0:
            ax.plot(self.historial['epoch'], self.historial['train_r2'], label='Train R2', color="#3681c7", linestyle='--', linewidth=2)
            ax.plot(self.historial['epoch'], self.historial['test_r2'], label='Test R2', color="#e9944f", linewidth=2)
            ax.set_xlabel('Epoca')
            ax.set_ylabel('R2')
            ax.set_title('Evolucion de la Precisión')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def finalizar(self):
        '''
        Finalización del entrenamiento.
        '''
        elapsed = time.time() - self.inicio
        if elapsed < 60:
            tiempo_total = f"{elapsed:.1f} segundos"
        elif elapsed < 3600:
            tiempo_total = f"{elapsed/60:.1f} minutos"
        else:
            tiempo_total = f"{elapsed/3600:.1f} horas"
        
        print(f"Entrenamiento completado: {self.nombre_modelo}")
        print(f"Tiempo total: {tiempo_total}")
        print(f"Mejor Test Loss: {self.mejor_loss:.6f}")
        if len(self.historial['test_r2']) > 0:
            print(f"R2 Test final: {self.historial['test_r2'][-1]:.4f}")
            
        if self.guardar_graficas:
            self._guardar_grafica() 
            
        plt.ioff()
        plt.show()
        
        return self.historial

# Red Neuronal
class GeocronologiaNet(nn.Module):
    '''
    Red neuronal para hacer estimación de edades geológicas.
    Aprende a predecir la edad en base a las concentraciones de isótopos padres e hijos,
    relacionándolos y tomando las características del sistema.
    Las capas completamente conectadas se dan por BatchNormalization, se toma activación por LeakyReLU
    para gradientes negativos pequeños, con menor dropout en capas más rofundas.
    '''
    def __init__(self, input_dim=2, hidden_dims=[128, 64, 32], output_dim=1, dropout_rate=0.3):
        '''
        Inicialización de la red neuronal.
        input_dim: Número de características de entrada.
        hidden_dims: Lista con el tamaño de cada capa oculta.
        output_dim: Número de neuronas de salida.
        dropout_rate: Tasa de dropout inicial. 
        '''
        super(GeocronologiaNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate * (1 - i/len(hidden_dims))))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        '''
        Inicialización de los pesos por medio de Kaiming debido a quepermite mantener 
        gradientes en rangos adecuados, lo que mejora la convergencia en redes profundas.
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu') # Considerar solo entrada
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        '''
        Propagación hacia adelante.
        x: Datos de entrada
        Regresa las predicciones por torch.Tensor
        '''
        return self.network(x)

# Entrenamiento
def entrenar_red(modelo, X_train, y_train, X_test, y_test, epochs=500, batch_size=128, lr=0.0005, patience=60, device=device, nombre_modelo="Red Neuronal", scaler_y=None):
    '''
    Entrenamiento de la red neuronal con visualización actualizada.
    modelo: Red neuronal a entrenar.
    X_train: Datos de entrenamiento.
    y_train: Etiquetas de entrenamiento.
    X_test: Datos de validación .
    y_test: Etiquetas de validación.
    epochs: Número máximo de épocas.
    batch_size: Tamaño del lote.
    lr: Tasa de aprendizaje inicial.
    patience: Épocas sin mejora antes de early stopping.
    nombre_modelo: Nombre para visualización.
    scaler_y: Escalador para desescalar predicciones.
    '''
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Mezcla los datos de cada época para generalizar
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    modelo.to(device)
    
    visualizador = EntrenamientoVisualizador(epochs, nombre_modelo)
    visualizador.iniciar()
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        modelo.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = modelo(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        
        modelo.eval()
        with torch.no_grad():
            y_pred_train = modelo(X_train_t)
            y_pred_test = modelo(X_test_t)
            test_loss = criterion(y_pred_test, y_test_t).item()
        
        if scaler_y is not None:
            y_train_true = scaler_y.inverse_transform(y_train_t.cpu().numpy().reshape(-1, 1))
            y_train_pred = scaler_y.inverse_transform(y_pred_train.cpu().numpy().reshape(-1, 1))
            y_test_true = scaler_y.inverse_transform(y_test_t.cpu().numpy().reshape(-1, 1))
            y_test_pred = scaler_y.inverse_transform(y_pred_test.cpu().numpy().reshape(-1, 1))
            
            train_r2 = r2_score(y_train_true, y_train_pred)
            test_r2 = r2_score(y_test_true, y_test_pred)
        else:
            train_r2 = r2_score(y_train_t.cpu().numpy(), y_pred_train.cpu().numpy())
            test_r2 = r2_score(y_test_t.cpu().numpy(), y_pred_test.cpu().numpy())
        
        scheduler.step()
        
        mejorado = test_loss < best_loss
        if mejorado:
            best_loss = test_loss
            patience_counter = 0
            best_state = modelo.state_dict()
        else:
            patience_counter += 1
        
        visualizador.actualizar(epoch, epoch_loss, test_loss, train_r2, test_r2, mejorado)
        
        if patience_counter >= patience:
            print(f"\nEarly stopping en epoca {epoch+1} (paciencia: {patience})")
            break
    
    if best_state is not None:
        modelo.load_state_dict(best_state)
    
    visualizador.finalizar()
    return modelo, visualizador.historial

# Cargar datos
def limpiar_ruta(ruta):
    """
    Limpia la ruta eliminando comillas y espacios.
    ruta: Ruta del archivo.
    """
    ruta = ruta.strip()
    if (ruta.startswith('"') and ruta.endswith('"')) or (ruta.startswith("'") and ruta.endswith("'")):
        ruta = ruta[1:-1]
    return ruta

def cargar_datos_excel(ruta_excel):
    """
    Carga datos del Excel.
    Tomando como referencia el archivo adjunto con datos de zircón.
    """
    try:
        ruta_excel = limpiar_ruta(ruta_excel)
        
        if not os.path.exists(ruta_excel):
            print(f"  Archivo no encontrado: {ruta_excel}")
            return None
        
        df_zircon = pd.read_excel(
            ruta_excel, 
            sheet_name='Table S3',
            header=1,
            thousands=None,
            decimal=','
        )
        
        print(f"  Datos cargados: {len(df_zircon)} muestras de zircon")
        print(f"  Columnas encontradas: {len(df_zircon.columns)}")
        
        return df_zircon
    except Exception as e:
        print(f"  Error al cargar Excel: {e}")
        return None

def procesar_datos_reales(df):
    """
    Procesa datos reales para entrenamiento. Identifica la columna de edad, selecciona las
    características más erlevantes y elimina filas con valores nulos.
    """
    if df is None:
        return None
    
    edad_col = None
    for col in df.columns:
        if '207Pb/206Pb age' in col and '(Ma)' in col:
            edad_col = col
            break
        elif 'age' in col.lower() and '(Ma)' in col:
            edad_col = col
            break
    
    if edad_col is None:
        edad_col = '207Pb/206Pb age (Ma)'
        if edad_col not in df.columns:
            return None
    
    features_reales = []
    cols_numericas = ['Th (ppm)', 'U (ppm)', 'Pb (total, ppm)', 'Th/U', 
                      'SiO2 (%)', 'Y', 'Nb', 'Hf', 'Ta', 'W']
    
    for col in cols_numericas:
        if col in df.columns:
            features_reales.append(col)
    
    if not features_reales:
        features_reales = df.select_dtypes(include=[np.number]).columns.tolist()
        features_reales = [c for c in features_reales if c != edad_col]
        features_reales = features_reales[:10]
    
    print(f"  Caracteristicas: {len(features_reales)}")
    
    df_procesado = df[features_reales + [edad_col]].copy()
    df_procesado = df_procesado.dropna()
    
    for col in features_reales:
        if df_procesado[col].dtype in ['float64', 'int64']:
            Q1 = df_procesado[col].quantile(0.01)
            Q3 = df_procesado[col].quantile(0.99)
            IQR = Q3 - Q1
            if IQR > 0: # Filtración de outliers en caso de variaciones
                df_procesado = df_procesado[
                    (df_procesado[col] >= Q1 - 1.5*IQR) & 
                    (df_procesado[col] <= Q3 + 1.5*IQR)
                ]
    
    print(f"  Muestras despues de limpieza: {len(df_procesado)}")
    return df_procesado, features_reales, edad_col

def generar_datos_sinteticos(n_muestras=3000, semilla=42):
    """
    Genera datos sinteticos cuando no hay datos reales. Se simula el decaimiento de los sistemas
    en base a su vida media. Los datos generados tienen decaimiento exponencial con ruido gaussiano
    y de Poisson, edades con distribución multimodal por eventos geológicos y concentraciones iniciales.
    n_muestras: Número total de muestras a generar.
    semilla: Semilla para reproducir.
    """
    np.random.seed(semilla)
    
    sistemas = [
        {'nombre': 'U235', 'lambda': 0.693/703.8, 'vida_media': 703.8},
        {'nombre': 'U238', 'lambda': 0.693/4468.0, 'vida_media': 4468.0},
        {'nombre': 'K40_Ar', 'lambda': 0.693/11930.0, 'vida_media': 11930.0},
        {'nombre': 'K40_Ca', 'lambda': 0.693/1396.0, 'vida_media': 1396.0},
        {'nombre': 'Rb87', 'lambda': 0.693/48800.0, 'vida_media': 48800.0},
        {'nombre': 'Sm147', 'lambda': 0.693/106000.0, 'vida_media': 106000.0}
    ]
    
    n_por_sistema = n_muestras // len(sistemas)
    n_restante = n_muestras - (n_por_sistema * len(sistemas))
    
    dfs = []
    
    for i, sistema in enumerate(sistemas):
        n_actual = n_por_sistema + (n_restante if i == len(sistemas)-1 else 0)
        
        n1 = n_actual // 3 # Evento temprano
        n2 = n_actual // 3 # Evento medio
        n3 = n_actual - n1 - n2 # Evento tardío
        
        t = np.concatenate([ #Edades en millones de años
            np.random.normal(500, 200, n1), # Edad jóven 
            np.random.normal(2500, 500, n2), # Edad media
            np.random.normal(4000, 300, n3) # Edad antigua
        ])
        t = np.clip(t, 0, 5000)
        np.random.shuffle(t)
        
        N0 = np.random.lognormal(mean=13, sigma=1.5, size=n_actual)
        
        N_verdadero = N0 * np.exp(-sistema['lambda'] * t)
        D_verdadero = N0 * (1 - np.exp(-sistema['lambda'] * t))
        
        ruido_sigma = 0.03 + 0.02 * np.random.rand(n_actual)
        ruido_N = ruido_sigma * N_verdadero * np.random.randn(n_actual)
        ruido_D = ruido_sigma * D_verdadero * np.random.randn(n_actual)
        
        mask_poisson = np.random.rand(n_actual) < 0.2
        ruido_N[mask_poisson] += np.sqrt(np.maximum(N_verdadero[mask_poisson], 1)) * np.random.randn(np.sum(mask_poisson))
        ruido_D[mask_poisson] += np.sqrt(np.maximum(D_verdadero[mask_poisson], 1)) * np.random.randn(np.sum(mask_poisson))
        
        N_medido = np.maximum(N_verdadero + ruido_N, 0)
        D_medido = np.maximum(D_verdadero + ruido_D, 0)
        
        df = pd.DataFrame({
            't': t,
            'N0': N0,
            'N_padre': N_medido,
            'D_hijo': D_medido,
            'lambda_real': sistema['lambda'],
            'vida_media': sistema['vida_media'],
            'sistema': sistema['nombre']
        })
        
        dfs.append(df)
        print(f"  {sistema['nombre']}: {n_actual} muestras")
    
    df_total = pd.concat(dfs, ignore_index=True)
    df_total = df_total[df_total['N_padre'] > 0]
    df_total = df_total[df_total['D_hijo'] > 0]
    
    print(f"  Total: {len(df_total)} muestras sinteticas")
    return df_total

# Funciones de evaluación
def evaluar_con_cross_validation(modelo, X, y, cv=5, nombre="Modelo", scaler_y=None):
    """
    Evalua un modelo usando Cross Validation K-Fold. Dada la división de los datos para validación y
    entrenamiento, es posible tene runa estimación del rendimiento del modelo.
    X: Características de entrada.
    y: Etiquetas objetivo.
    cv: Número de parámetros para la validación cruzada.
    nombre: Nombre del modelo para identificación en prints.
    scaler_y: Escalador para desescalar predicciones.
    """
    print(f"\n  {nombre} - Cross Validation (K={cv}):")

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    scores_r2 = []
    scores_rmse = []
    scores_mae = []
    scores_mape = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_f, X_val_f = X[train_idx], X[val_idx]
        y_train_f, y_val_f = y[train_idx], y[val_idx]
        
        from sklearn.base import clone
        modelo_fold = clone(modelo)
        modelo_fold.fit(X_train_f, y_train_f)
        y_pred_f = modelo_fold.predict(X_val_f)
        
        if scaler_y is not None:
            y_true_f = scaler_y.inverse_transform(y_val_f.reshape(-1, 1))
            y_pred_f = scaler_y.inverse_transform(y_pred_f.reshape(-1, 1))
        else:
            y_true_f = y_val_f.reshape(-1, 1)
            y_pred_f = y_pred_f.reshape(-1, 1)
        
        r2 = r2_score(y_true_f, y_pred_f)
        rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
        mae = mean_absolute_error(y_true_f, y_pred_f)
        mape = np.mean(np.abs((y_true_f - y_pred_f) / (y_true_f + 1e-10))) * 100
        
        scores_r2.append(r2)
        scores_rmse.append(rmse)
        scores_mae.append(mae)
        scores_mape.append(mape)
        
        print(f"    Fold {fold+1}: R2={r2:.4f}, RMSE={rmse:.2f} Ma")
    
    r2_mean = np.mean(scores_r2)
    r2_std = np.std(scores_r2)
    rmse_mean = np.mean(scores_rmse)
    rmse_std = np.std(scores_rmse)
    mae_mean = np.mean(scores_mae)
    mape_mean = np.mean(scores_mape)
    
    print(f"\n    Resultados CV:")
    print(f"      R2:  {r2_mean:.4f} +/- {r2_std:.4f}")
    print(f"      RMSE: {rmse_mean:.2f} +/- {rmse_std:.2f} Ma")
    print(f"      MAE:  {mae_mean:.2f} +/- {np.std(scores_mae):.2f} Ma")
    print(f"      MAPE: {mape_mean:.2f}% +/- {np.std(scores_mape):.2f}%")
    
    return {
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'mae_mean': mae_mean,
        'mape_mean': mape_mean,
        'scores_r2': scores_r2,
        'scores_rmse': scores_rmse
    }

def evaluar_modelo_con_cv(modelo, X_train, y_train, X_test, y_test, scaler_y=None, nombre="Modelo", cv=5):
    """
    Evalua un modelo con entrenamiento en train, evaluación en test y Cross Validation. Se entrena entrena
    el modelo con el conjunto de entrenamiento, después se evalúa con el conjunto de teste, se realiza
    cv en este y se retornan las métricas.
    cv: Número de folds para validación cruzada.
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    if scaler_y is not None:
        y_true_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        y_pred_test = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    else:
        y_true_test = y_test.reshape(-1, 1)
        y_pred_test = y_pred.reshape(-1, 1)
    
    r2_test = r2_score(y_true_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    mae_test = mean_absolute_error(y_true_test, y_pred_test)
    mape_test = np.mean(np.abs((y_true_test - y_pred_test) / (y_true_test + 1e-10))) * 100
    
    print(f"\n  {nombre} - Evaluacion en Test:")
    print(f"    R2:   {r2_test:.4f}")
    print(f"    RMSE: {rmse_test:.2f} Ma")
    print(f"    MAE:  {mae_test:.2f} Ma")
    print(f"    MAPE: {mape_test:.2f}%")
    
    cv_results = evaluar_con_cross_validation(
        modelo, X_train, y_train, cv=cv, 
        nombre=nombre, scaler_y=scaler_y
    )
    
    return {
        'test': {'r2': r2_test, 'rmse': rmse_test, 'mae': mae_test, 'mape': mape_test},
        'cv': cv_results,
        'y_true': y_true_test,
        'y_pred': y_pred_test
    }

# Graficas de funciones
def graficar_comparativa(resultados, titulo="Comparacion de Modelos", mostrar_cv=True):
    """
    Grafica comparativa de modelos. Si mostrar_cv=True, muestra Test + CV.
    Si mostrar_cv=False, muestra solo Test.
    """
    # Filtrar modelos con datos validos
    modelos_validos = []
    for m in resultados.keys():
        if isinstance(resultados[m], dict) and 'test' in resultados[m]:
            if resultados[m]['test']['r2'] is not None:
                modelos_validos.append(m)
    
    if not modelos_validos:
        print("  No hay modelos validos para graficar.")
        return
    
    # Si mostrar_cv=True, filtrar solo modelos con CV valido
    if mostrar_cv:
        modelos_con_cv = []
        for m in modelos_validos:
            if m in resultados and resultados[m]['cv'] is not None:
                cv_data = resultados[m]['cv']
                if cv_data.get('r2_mean') is not None:
                    modelos_con_cv.append(m)
        
        if modelos_con_cv:
            modelos = modelos_con_cv
        else:
            print("  No hay modelos con CV valido. Mostrando solo Test.")
            mostrar_cv = False
            modelos = modelos_validos
    else:
        modelos = modelos_validos
    
    # Preparar datos
    r2_test = [resultados[m]['test']['r2'] for m in modelos]
    rmse_test = [resultados[m]['test']['rmse'] for m in modelos]
    mape_test = [resultados[m]['test']['mape'] for m in modelos]
    
    # Datos de CV (si estan disponibles)
    r2_cv = []
    r2_cv_std = []
    rmse_cv = []
    rmse_cv_std = []
    mape_cv = []
    tiene_cv = False
    
    for m in modelos:
        if m in resultados and resultados[m]['cv'] is not None:
            cv_data = resultados[m]['cv']
            if cv_data.get('r2_mean') is not None:
                r2_cv.append(cv_data['r2_mean'])
                r2_cv_std.append(cv_data.get('r2_std', 0))
                rmse_cv.append(cv_data.get('rmse_mean', 0))
                rmse_cv_std.append(cv_data.get('rmse_std', 0))
                mape_cv.append(cv_data.get('mape_mean', 0))
                tiene_cv = True
            else:
                r2_cv.append(0)
                r2_cv_std.append(0)
                rmse_cv.append(0)
                rmse_cv_std.append(0)
                mape_cv.append(0)
        else:
            r2_cv.append(0)
            r2_cv_std.append(0)
            rmse_cv.append(0)
            rmse_cv_std.append(0)
            mape_cv.append(0)
    
    # Si no hay CV real, mostrar solo Test
    if not tiene_cv and mostrar_cv:
        mostrar_cv = False
    
    # Determinar numero de graficas
    if mostrar_cv and tiene_cv:
        n_plots = 3
        titulo_plot = titulo
    else:
        n_plots = 2
        titulo_plot = f"{titulo} - Solo Test"
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle(titulo_plot, fontsize=14, fontweight='bold')
    
    x = np.arange(len(modelos))
    width = 0.35
    idx = 0
    
    # Grafica 1: R²
    ax = axes[idx]
    if mostrar_cv and tiene_cv:
        ax.bar(x - width/2, r2_test, width, label='Test', color='steelblue')
        ax.bar(x + width/2, r2_cv, width, label='CV', color='orange', 
               yerr=r2_cv_std, capsize=3)
    else:
        ax.bar(x, r2_test, width, color='steelblue', label='Test')
    
    ax.set_xlabel('Modelo')
    ax.set_ylabel('R²')
    ax.set_title('Comparacion de R²')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=45, ha='right')
    ax.axhline(y=0.85, color="#149152", linestyle='--', alpha=0.5, label='Bueno (0.85)')
    ax.axhline(y=0.70, color="#af5010", linestyle='--', alpha=0.5, label='Aceptable (0.70)')
    idx += 1
    
    # Grafica 2: RMSE
    ax = axes[idx]
    if mostrar_cv and tiene_cv:
        ax.bar(x - width/2, rmse_test, width, label='Test', color='steelblue')
        ax.bar(x + width/2, rmse_cv, width, label='CV', color='orange',
               yerr=rmse_cv_std, capsize=3)
    else:
        ax.bar(x, rmse_test, width, color='steelblue', label='Test')
    
    ax.set_xlabel('Modelo')
    ax.set_ylabel('RMSE (Ma)')
    ax.set_title('Comparacion de RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=45, ha='right')
    idx += 1
    
    # Grafica 3: MAPE (solo si hay CV y mostrar_cv=True)
    if mostrar_cv and tiene_cv and n_plots == 3:
        ax = axes[idx]
        ax.bar(x - width/2, mape_test, width, label='Test', color='steelblue')
        ax.bar(x + width/2, mape_cv, width, label='CV', color='orange')
        ax.set_xlabel('Modelo')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('Comparacion de MAPE')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def graficar_analisis_completo(y_true, y_pred, titulo="Analisis de Prediccion"):
    """
    Análisas de predicciones de dispersión, distribución, error relativo y erro por rango de edad.
    y_true: Valores reales.
    y_pred: Valores predichos.
    titulo: Título de la figura.
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    errores = y_true.flatten() - y_pred.flatten()
    errores_rel = np.abs(errores / (y_true.flatten() + 1e-10)) * 100
    errores_rel = np.clip(errores_rel, 0, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.4, s=8, c="#3681c7")
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'r--', lw=2, label='Perfecta')
    ax.set_xlabel("Edad Real (Ma)")
    ax.set_ylabel("Edad Predicha (Ma)")
    ax.set_title(f"Prediccion vs Real\nR2 = {r2_score(y_true, y_pred):.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(errores, bins=40, alpha=0.7, color="#3681c7", edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', label='Error cero')
    ax.axvline(x=np.mean(errores), color="#e9944f", linestyle='--', 
               label=f'Media: {np.mean(errores):.1f}')
    ax.axvline(x=np.median(errores), color='green', linestyle='--', 
               label=f'Mediana: {np.median(errores):.1f}')
    ax.set_xlabel("Error (Real - Predicho)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion de Errores")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.scatter(y_true, errores_rel, alpha=0.4, s=8, c='coral')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10%')
    ax.axhline(y=20, color="#e9944f", linestyle='--', alpha=0.7, label='20%')
    ax.set_xlabel("Edad Real (Ma)")
    ax.set_ylabel("Error Relativo (%)")
    ax.set_title("Error Relativo vs Edad")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    rangos = [0, 1000, 2000, 3000, 4000, 5000]
    etiquetas = [f"{rangos[i]}-{rangos[i+1]}" for i in range(len(rangos)-1)]
    rmse_por_rango = []
    mape_por_rango = []
    n_por_rango = []
    
    for i in range(len(rangos)-1):
        mask = (y_true.flatten() >= rangos[i]) & (y_true.flatten() < rangos[i+1])
        if np.sum(mask) > 0:
            rmse_por_rango.append(np.sqrt(np.mean(errores[mask]**2)))
            mape_por_rango.append(np.mean(errores_rel[mask]))
            n_por_rango.append(np.sum(mask))
        else:
            rmse_por_rango.append(0)
            mape_por_rango.append(0)
            n_por_rango.append(0)
    
    x_pos = np.arange(len(etiquetas))
    width = 0.35
    ax.bar(x_pos - width/2, rmse_por_rango, width, label='RMSE (Ma)', color="#3681c7")
    ax2 = ax.twinx()
    ax2.bar(x_pos + width/2, mape_por_rango, width, label='MAPE (%)', color='coral', alpha=0.7)
    
    for i, (rmse, mape, n) in enumerate(zip(rmse_por_rango, mape_por_rango, n_por_rango)):
        if rmse > 0:
            ax.text(i - width/2, rmse + 50, f'n={n}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Rango de Edad (Ma)')
    ax.set_ylabel('RMSE (Ma)')
    ax2.set_ylabel('MAPE (%)')
    ax.set_title('Error por Rango de Edad')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()

# Menú
def main():
    print("Geocronología")
    print("Analisis de datos isotopicos con Machine Learning")

    print("Seleccione una pción:")
    print("  1. Datos REALES + RBM")
    print("  2. Datos SINTETICOS (solo simulacion)")
    
    while True:
        try:
            opcion = input("\nIngrese su opcion: ").strip()
            if opcion in ['1', '2']:
                break
            else:
                print("  Opcion invalida. Ingrese 1 o 2.")
        except:
            print("  Entrada invalida.")
    
    usar_rbm = (opcion == '1')
    
    # Opción 1
    if usar_rbm:
        print("Opción 1: Datos REALES + RBM")
        
        print("\nIngrese la ruta del archivo Excel con los datos:")
        print("  (Ejemplo: C:\\Users\\usuario\\documento.xlsx)")
        print("  (Si presiona Enter, buscara automaticamente)")
        
        ruta_excel = input("\nRuta: ").strip()
        
        if not ruta_excel:
            archivos = [f for f in os.listdir('.') if f.endswith('.xlsx') and 'mmc2' in f]
            if archivos:
                ruta_excel = archivos[0]
                print(f"  Archivo encontrado: {ruta_excel}")
            else:
                print("  No se encontro archivo Excel. Cambiando a Opción 2")
                usar_rbm = False
        
        if usar_rbm:
            print("\nCargando datos del Excel")
            df_raw = cargar_datos_excel(ruta_excel)
            if df_raw is None:
                print("  Error al cargar datos. Cambiando a Opcion 2")
                usar_rbm = False
        
        if usar_rbm:
            print("\nProcesando datos reales")
            resultado = procesar_datos_reales(df_raw)
            if resultado is None:
                print("  Error al procesar datos. Cambiando a Opcion 2")
                usar_rbm = False
            else:
                df_real, features_reales, edad_col = resultado
        
        if usar_rbm:
            print(f"\n  Datos reales cargados: {len(df_real)} muestras")
            print(f"  Rango de edades: {df_real[edad_col].min():.0f} - {df_real[edad_col].max():.0f} Ma")
            print(f"  Edad media: {df_real[edad_col].mean():.0f} Ma")
            
            print("\nPreparando datos para RBM")
            
            X_real = df_real[features_reales].values.astype(np.float32)
            y_real = df_real[edad_col].values.astype(np.float32)
            
            imputer = SimpleImputer(strategy='median')
            X_real = imputer.fit_transform(X_real)
            
            scaler_rbm = MinMaxScaler()
            X_rbm = scaler_rbm.fit_transform(X_real)
            X_rbm_bin = (X_rbm > 0.5).astype(np.float32)
            
            print(f"  Caracteristicas: {len(features_reales)}")
            print(f"  Muestras para RBM: {len(X_rbm_bin)}")
            
            # Configurar RBM
            print("Configuración RBM")
            print("  1. Cargar RBM guardada desde archivo.")
            print("  2. Entrenar RBM desde cero.")
            print("  3. Entrenar RBM con parametros predefinidos.")
            
            while True:
                try:
                    opcion_rbm = input("\nIngrese su opción: ").strip()
                    if opcion_rbm in ['1', '2', '3']:
                        break
                    else:
                        print("  Opcion invalida. Ingrese 1, 2 o 3.")
                except:
                    print("  Entrada invalida.")
            
            rbm = None
            rbm_entrenada = False
            losses_rbm = None
            
            # Opción 1
            if opcion_rbm == '1':
                print("\n" + "="*70)
                print("CARGANDO RBM GUARDADA")
                print("="*70)
                
                import glob
                archivos_pth = glob.glob("*.pth")
                if archivos_pth:
                    print("\nArchivos .pth encontrados:")
                    for i, archivo in enumerate(archivos_pth, 1):
                        print(f"  {i}. {archivo}")
                    print(f"  {len(archivos_pth)+1}. Ingresar ruta manualmente")
                    
                    seleccion = input("\nSeleccione un archivo (numero): ").strip()
                    try:
                        idx = int(seleccion) - 1
                        if 0 <= idx < len(archivos_pth):
                            ruta_archivo = archivos_pth[idx]
                        else:
                            ruta_archivo = input("Ingrese la ruta del archivo .pth: ").strip()
                    except:
                        ruta_archivo = input("Ingrese la ruta del archivo .pth: ").strip()
                else:
                    ruta_archivo = input("Ingrese la ruta del archivo .pth: ").strip()
                    if not ruta_archivo:
                        ruta_archivo = "rbm_entrenada.pth"
                
                try:
                    rbm = RBMGenerador.cargar_modelo(ruta_archivo)
                    print(f"  RBM cargada desde: {ruta_archivo}")
                    rbm_entrenada = True
                except FileNotFoundError:
                    print(f"  Archivo no encontrado: {ruta_archivo}")
                    print("  ¿Desea entrenar una RBM desde cero?")
                    respuesta = input("  (s/n): ").strip().lower()
                    if respuesta == 's':
                        opcion_rbm = '2'
                    else:
                        print("  No se pudo cargar la RBM.")
                        return None
                except Exception as e:
                    print(f"  Error al cargar la RBM: {e}")
                    print("  ¿Desea entrenar una RBM desde cero?")
                    respuesta = input("  (s/n): ").strip().lower()
                    if respuesta == 's':
                        opcion_rbm = '2'
                    else:
                        print("  No se pudo cargar la RBM.")
                        return None
            
            # # Opción 2
            if opcion_rbm == '2':
                print("Entrenando RBM desde cero")
                
                print("\nConfiguracion de entrenamiento (presione Enter para usar valores por defecto):")
                
                n_hidden_input = input(f"  Neuronas ocultas (default: {len(features_reales)//2}): ").strip()
                n_hidden = int(n_hidden_input) if n_hidden_input else max(len(features_reales) // 2, 10)
                
                epochs_input = input("  Epocas (default: 80): ").strip()
                epochs = int(epochs_input) if epochs_input else 80
                
                batch_size_input = input("  Batch size (default: 32): ").strip()
                batch_size = int(batch_size_input) if batch_size_input else 32
                
                lr_input = input("  Learning rate (default: 0.01): ").strip()
                lr = float(lr_input) if lr_input else 0.01
                
                temperatura_input = input("  Temperatura (default: 0.8): ").strip()
                temperatura = float(temperatura_input) if temperatura_input else 0.8
                
                print("\nCreando RBM")
                rbm = RBMGenerador(
                    n_visible=len(features_reales),
                    n_hidden=n_hidden,
                    temperatura=temperatura
                )
                
                print("\nEntrenando RBM")
                losses_rbm = rbm.entrenar(
                    X_rbm_bin,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    k=1,
                    nombre="RBM Geocronologia"
                )
                
                guardar = input("\n¿Desea guardar la RBM entrenada? (s/n): ").strip().lower()
                if guardar == 's':
                    ruta_guardar = input("  Ruta para guardar (default: rbm_entrenada.pth): ").strip()
                    if not ruta_guardar:
                        ruta_guardar = "rbm_entrenada.pth"
                    rbm.guardar_modelo(ruta_guardar)
                
                rbm_entrenada = True
            
            # # Opción 3
            if opcion_rbm == '3':
                print("Entrenando con parámetros definidos")
                print("  Neuronas ocultas: {} (mitad de las visibles)".format(len(features_reales)//2))
                print("  Epocas: 80")
                print("  Batch size: 32")
                print("  Learning rate: 0.01")
                print("  Temperatura: 0.8")
                
                n_hidden = max(len(features_reales) // 2, 10)
                
                print("\nCreando RBM")
                rbm = RBMGenerador(
                    n_visible=len(features_reales),
                    n_hidden=n_hidden,
                    temperatura=0.8
                )
                
                print("\nEntrenando RBM")
                losses_rbm = rbm.entrenar(
                    X_rbm_bin,
                    epochs=80,
                    batch_size=32,
                    lr=0.01,
                    k=1,
                    nombre="RBM Geocronologia"
                )
                
                rbm.guardar_modelo("rbm_entrenada.pth")
                print("\n  RBM guardada automaticamente como 'rbm_entrenada.pth'")
                
                rbm_entrenada = True
            
            if not rbm_entrenada:
                print("  No se pudo configurar la RBM.")
                return None
            
            print("\nGenerando datos sinteticos con RBM")
            
            n_generar = max(len(X_rbm_bin) * 2, 2000)
            print(f"  Generando {n_generar} muestras sinteticas")
            muestras_rbm = rbm.generar_muestras(n_generar, n_steps=150, semilla=42)
            X_gen = scaler_rbm.inverse_transform(muestras_rbm)
            
            y_gen = np.random.choice(y_real, len(X_gen))
            y_gen = y_gen + np.random.normal(0, 50, len(X_gen))
            y_gen = np.clip(y_gen, y_real.min(), y_real.max())
            
            df_gen = pd.DataFrame(X_gen, columns=features_reales)
            df_gen[edad_col] = y_gen
            
            print(f"  Datos generados: {len(df_gen)} muestras")
            
            print("\nPreparando datos combinados (reales + RBM)")
            
            X_combined = np.vstack([X_real, X_gen])
            y_combined = np.concatenate([y_real, y_gen])
            
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()
            
            X_scaled = scaler_X.fit_transform(X_combined)
            y_scaled = scaler_y.fit_transform(y_combined.reshape(-1, 1)).ravel()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            print(f"  Train: {X_train.shape[0]} muestras")
            print(f"  Test: {X_test.shape[0]} muestras")
            
            print("\nEntrenando modelos con Cross Validation")
            
            modelos = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=0.01, max_iter=1000),
                'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000),
                'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=15, 
                                                      random_state=42, n_jobs=-1),
                'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
                'MLP (Sklearn)': MLPRegressor(hidden_layer_sizes=(200, 100, 50), 
                                             max_iter=500, random_state=42, early_stopping=True)
            }
            
            resultados = {}
            resultados_cv = {}
            
            for nombre, modelo in modelos.items():
                print(f"\n  {nombre}:")
                resultado = evaluar_modelo_con_cv(
                    modelo, X_train, y_train, X_test, y_test,
                    scaler_y=scaler_y, nombre=nombre, cv=5
                )
                resultados[nombre] = resultado
                resultados_cv[nombre] = resultado['cv']
            
            print("\n  Entrenando Red Neuronal")
            modelo_nn = GeocronologiaNet(input_dim=X_train.shape[1], hidden_dims=[256, 128, 64, 32])
            modelo_nn, historial = entrenar_red( modelo_nn, X_train, y_train.reshape(-1, 1),X_test, y_test.reshape(-1, 1), epochs=500, batch_size=128, lr=0.0005, patience=60, device=device, nombre_modelo="Red Neuronal + RBM", scaler_y=scaler_y )
            
            with torch.no_grad():
                y_pred_nn = modelo_nn(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
            y_true_nn = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            y_pred_nn = scaler_y.inverse_transform(y_pred_nn)
            
            r2_nn = r2_score(y_true_nn, y_pred_nn)
            rmse_nn = np.sqrt(mean_squared_error(y_true_nn, y_pred_nn))
            mae_nn = mean_absolute_error(y_true_nn, y_pred_nn)
            mape_nn = np.mean(np.abs((y_true_nn - y_pred_nn) / (y_true_nn + 1e-10))) * 100
            
            resultados['Red Neuronal'] = {
                'test': {'r2': r2_nn, 'rmse': rmse_nn, 'mae': mae_nn, 'mape': mape_nn},
                'cv': {'r2_mean': None, 'r2_std': None, 'rmse_mean': None, 'rmse_std': None, 'mae_mean': None, 'mape_mean': None},
                'y_true': y_true_nn,
                'y_pred': y_pred_nn
            }
            
            print("\n Generando graficas finales")
            
            graficar_comparativa(resultados, titulo="Comparacion de Modelos - Datos Reales + RBM", mostrar_cv=True)
            graficar_comparativa(resultados, titulo="Cross Validation - Datos Reales + RBM", mostrar_cv=False)
            
            mejor_modelo = max(resultados, key=lambda x: resultados[x]['test']['r2'] if resultados[x]['test']['r2'] is not None else -999)
            print(f"\nMejor modelo: {mejor_modelo}")
            print(f"  R2 Test: {resultados[mejor_modelo]['test']['r2']:.4f}")
            print(f"  RMSE Test: {resultados[mejor_modelo]['test']['rmse']:.2f} Ma")
            
            graficar_analisis_completo(resultados[mejor_modelo]['y_true'], resultados[mejor_modelo]['y_pred'], titulo=f"Analisis de Prediccion - {mejor_modelo}")
            
            print(f"\nDatos reales: {len(df_real)} muestras")
            print(f"Datos generados por RBM: {len(df_gen)} muestras")
            print(f"Total combinado: {len(X_combined)} muestras")
            
            print(f"\nRBM:")
            print(f"  Neuronas visibles: {rbm.n_visible}")
            print(f"  Neuronas ocultas: {rbm.n_hidden}")
            print(f"  Temperatura: {rbm.temperatura}")
            
            print("\nRanking de modelos (Test R2):")
            sorted_models = sorted(resultados.items(), key=lambda x: x[1]['test']['r2'] if x[1]['test']['r2'] is not None else -999, reverse=True)
            for i, (nombre, metricas) in enumerate(sorted_models, 1):
                cv_str = ""
                if metricas['cv']['r2_mean'] is not None:
                    cv_str = f" | CV R2: {metricas['cv']['r2_mean']:.4f} +/- {metricas['cv']['r2_std']:.4f}"
                print(f"  {i}. {nombre:25} Test R2 = {metricas['test']['r2']:.4f} | RMSE = {metricas['test']['rmse']:.2f} Ma{cv_str}")
            
            return resultados
    
   # Opción 2
    print("Opción 2: Datos sintéticos")
    
    print("\n Generando datos sinteticos")
    df_sint = generar_datos_sinteticos(n_muestras=3000, semilla=42)
    
    print("\n Ingenieria de caracteristicas")
    
    df = df_sint.copy()
    df['D_N_ratio'] = df['D_hijo'] / (df['N_padre'] + 1e-10)
    df['log_N'] = np.log(df['N_padre'] + 1e-10)
    df['log_D'] = np.log(df['D_hijo'] + 1e-10)
    df['log_N0'] = np.log(df['N0'] + 1e-10)
    df['N_mas_D'] = df['N_padre'] + df['D_hijo']
    df['D_sobre_N_mas_D'] = df['D_hijo'] / (df['N_padre'] + df['D_hijo'] + 1e-10)
    df['N_sobre_N0'] = df['N_padre'] / (df['N0'] + 1e-10)
    df['D_sobre_N0'] = df['D_hijo'] / (df['N0'] + 1e-10)
    df['N_D_producto'] = df['N_padre'] * df['D_hijo']
    df['log_N_D_ratio'] = np.log(df['D_hijo'] / (df['N_padre'] + 1e-10) + 1)
    
    sistemas_encoded = pd.get_dummies(df['sistema'], prefix='sistema')
    df = pd.concat([df, sistemas_encoded], axis=1)
    df['log_vida_media'] = np.log(df['vida_media'] + 1e-10)
    df['inv_vida_media'] = 1 / (df['vida_media'] + 1e-10)
    
    features = ['N_padre', 'D_hijo', 'D_N_ratio', 'log_N', 'log_D', 
                'log_N0', 'N_mas_D', 'D_sobre_N_mas_D', 'N_sobre_N0', 
                'D_sobre_N0', 'N_D_producto', 'log_N_D_ratio',
                'log_vida_media', 'inv_vida_media'] + list(sistemas_encoded.columns)
    
    print(f"  Caracteristicas: {len(features)}")
    
    print("\n Preparando datos")
    
    X = df[features].values.astype(np.float32)
    y = df['t'].values.astype(np.float32)
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Test: {X_test.shape[0]} muestras")
    
    print("\nEntrenando modelos con Cross Validation")
    
    modelos = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01, max_iter=1000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=15, 
                                              random_state=42, n_jobs=-1),
        'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
        'MLP (Sklearn)': MLPRegressor(hidden_layer_sizes=(200, 100, 50), 
                                     max_iter=500, random_state=42, early_stopping=True)
    }
    
    resultados = {}
    resultados_cv = {}
    
    for nombre, modelo in modelos.items():
        print(f"\n  {nombre}:")
        resultado = evaluar_modelo_con_cv(
            modelo, X_train, y_train, X_test, y_test,
            scaler_y=scaler_y, nombre=nombre, cv=5
        )
        resultados[nombre] = resultado
        resultados_cv[nombre] = resultado['cv']
    
    print("\n  Entrenando Red Neuronal")
    modelo_nn = GeocronologiaNet(input_dim=X_train.shape[1], hidden_dims=[256, 128, 64, 32])
    modelo_nn, historial = entrenar_red(
        modelo_nn, X_train, y_train.reshape(-1, 1),
        X_test, y_test.reshape(-1, 1),
        epochs=500, batch_size=128, lr=0.0005, patience=60,
        device=device, nombre_modelo="Red Neuronal (Sinteticos)", scaler_y=scaler_y
    )
    
    with torch.no_grad():
        y_pred_nn = modelo_nn(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    y_true_nn = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_pred_nn = scaler_y.inverse_transform(y_pred_nn)
    
    r2_nn = r2_score(y_true_nn, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_true_nn, y_pred_nn))
    mae_nn = mean_absolute_error(y_true_nn, y_pred_nn)
    mape_nn = np.mean(np.abs((y_true_nn - y_pred_nn) / (y_true_nn + 1e-10))) * 100
    
    resultados['Red Neuronal (PyTorch)'] = {
        'test': {'r2': r2_nn, 'rmse': rmse_nn, 'mae': mae_nn, 'mape': mape_nn},
        'cv': {'r2_mean': None, 'r2_std': None, 'rmse_mean': None, 'rmse_std': None,
               'mae_mean': None, 'mape_mean': None},
        'y_true': y_true_nn,
        'y_pred': y_pred_nn
    }
    
    print("\nGenerando graficas finales")
    
    graficar_comparativa(resultados, titulo="Comparacion de Modelos - Datos Sinteticos", mostrar_cv=True)
    graficar_comparativa(resultados, titulo="Cross Validation - Datos Sinteticos", mostrar_cv=False)
    
    mejor_modelo = max(resultados, key=lambda x: resultados[x]['test']['r2'] if resultados[x]['test']['r2'] is not None else -999)
    print(f"\nMejor modelo: {mejor_modelo}")
    print(f"  R2 Test: {resultados[mejor_modelo]['test']['r2']:.4f}")
    print(f"  RMSE Test: {resultados[mejor_modelo]['test']['rmse']:.2f} Ma")
    
    graficar_analisis_completo(
        resultados[mejor_modelo]['y_true'],
        resultados[mejor_modelo]['y_pred'],
        titulo=f"Analisis de Prediccion - {mejor_modelo} (Sinteticos)"
    )
    
    print(f"\nDatos generados: {len(df)} muestras")
    
    print("\nRanking de modelos (Test R2):")
    sorted_models = sorted(resultados.items(), key=lambda x: x[1]['test']['r2'] if x[1]['test']['r2'] is not None else -999, reverse=True)
    for i, (nombre, metricas) in enumerate(sorted_models, 1):
        cv_str = ""
        if metricas['cv']['r2_mean'] is not None:
            cv_str = f" | CV R2: {metricas['cv']['r2_mean']:.4f} +/- {metricas['cv']['r2_std']:.4f}"
        print(f"  {i}. {nombre:25} Test R2 = {metricas['test']['r2']:.4f} | RMSE = {metricas['test']['rmse']:.2f} Ma{cv_str}")
    
    return resultados

if __name__ == "__main__":
    resultados = main()
