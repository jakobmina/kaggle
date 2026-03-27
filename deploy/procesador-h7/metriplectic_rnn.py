import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import logging
from tqdm import tqdm

# --- Autenticación Kaggle / Google Cloud ---
# Este bloque permite a Kaggle autenticar los servicios de GCP (ej. TPUs o GCS Buckets)
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    user_credential = user_secrets.get_gcloud_credential()
    user_secrets.set_tensorflow_credential(user_credential)
    print("✅ Kaggle Secrets & Google Cloud Authentication configurados exitosamente.")
except ImportError:
    # Ignorado silenciosamente en ejecución local
    pass

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- H7 Constantes Teóricas ---
PHI = (1 + math.sqrt(5)) / 2
PSI_1 = abs(math.cos(math.pi * PHI))
DRIFT_072 = 7 - 2 * math.pi      # ~ 0.7168
EPSILON = PSI_1 / 2              # ~ 0.1812 (Threshold del vacío)

# Pesos Metriplécticos Reales (71.7 - 28.3 - 1)
W_SYMP = DRIFT_072               # Conservación/Inercia (Base)
W_METR = 1.0 - DRIFT_072 - 0.01  # Disipación/Entropía
W_VAC  = 0.01

# --- Generador de Datos Holográficos Z_7 ---
def generate_basis(N=128, delta=1.0):
    B_obj = np.zeros((7, N))
    B_ref = np.zeros((7, N))
    for d in range(7):
        phi_d = PHI ** (d + 1)
        for n in range(N):
            n_val = n + 1
            B_obj[d, n] = math.cos(math.pi * phi_d * n_val + delta)
            B_ref[d, n] = math.cos(math.pi * phi_d * n_val - delta)
    return B_obj, B_ref

def ternary_collapse(H: np.ndarray, thresh: float) -> np.ndarray:
    T = np.zeros_like(H)
    T[H > thresh] = 1
    T[H < -thresh] = -1
    return T

class HolographicDataset(Dataset):
    """
    Dataset para el "H7 Attention Benchmark".
    Reconstrucción 128 -> 7 a partir de Epsilon Vacuum.
    """
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        self.B_obj, self.B_ref = generate_basis(128, delta=math.pi/4)
        
        self.T_data = []
        self.X_hat_data = []
        
        logger.info(f"Generando {num_samples} firmas cuánticas en Z7...")
        for _ in range(num_samples):
            # 1. State Original X (dim=7)
            x = np.random.randn(7)
            x = x / np.linalg.norm(x)
            
            # 2. Proyección Holográfica -> H (dim=128)
            H = x @ self.B_obj
            
            # 3. Colapso Ternario (La Entrada de la Red)
            T = ternary_collapse(H, EPSILON)
            
            # 4. Target de Reconstrucción (Ground Truth Real)
            X_hat = (H @ self.B_ref.T) / 128.0
            
            # Transformamos T (trits: -1, 0, 1) a indices (0, 1, 2)
            T_idx = T + 1
            
            self.T_data.append(torch.tensor(T_idx, dtype=torch.long))
            self.X_hat_data.append(torch.tensor(X_hat, dtype=torch.float32))

    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        return self.T_data[idx], self.X_hat_data[idx]

class GoldenModulationLayer(nn.Module):
    """
    Inyecta el Operador Áureo a la capa oculta.
    O_n = cos(pi * n) * cos(pi * phi * n)
    """
    def __init__(self):
        super(GoldenModulationLayer, self).__init__()
        
    def forward(self, x: torch.Tensor, step_n: int) -> torch.Tensor:
        O_n = math.cos(math.pi * step_n) * math.cos(math.pi * PHI * step_n)
        return x * O_n

class MetriplecticRNN(nn.Module):
    """
    Arquitectura para "The H7 Attention Task".
    Lee seq de 128 trits -> Reconstruye fase 7D.
    """
    def __init__(self, vocab_size: int = 3, embed_size: int = 16, hidden_size: int = 64, 
                 out_features: int = 7, num_layers: int = 2):
        super(MetriplecticRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.golden_modulation = GoldenModulationLayer()
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Salida: Reconstrucción 7D
        self.fc = nn.Linear(hidden_size, out_features)
    
    def forward(self, x: torch.Tensor, step_n: int):
        # x shape: (batch, 128)
        emb = self.embedding(x)  # (batch, 128, embed_size)
        
        # Aplicamos RNN
        rnn_out, (h_n, c_n) = self.rnn(emb) 
        
        # Extraemos el estado final de la secuencia para la predicción
        final_state = h_n[-1] # (batch, hidden_size)
        
        # Modulamos holográficamente
        modulated_out = self.golden_modulation(final_state, step_n)
        norm_out = self.layer_norm(modulated_out)
        
        # Proyectamos al vector de 7 dimensiones
        output = self.fc(norm_out)
        return output

class MetriplecticLoss(nn.Module):
    """
    Función de Pérdida Dual Regida por Constantes H7 (71.7% - 28.3% - 1%)
    Valida la Regla 1.1 y 1.2, resolviendo la Tarea de Métrica del Coseno.
    """
    def __init__(self, w_symp=W_SYMP, w_metr=W_METR, w_vac=W_VAC):
        super(MetriplecticLoss, self).__init__()
        self.w_symp = w_symp
        self.w_metr = w_metr
        self.w_vac = w_vac
        
    def forward(self, pred_7d: torch.Tensor, target_7d: torch.Tensor):
        # L_task evalúa qué tan mal está el vector resultante respecto al ground truth
        # CosineSimilarity(x, y) = [-1, 1], donde 1 es perfecto.
        # Loss debería ser 0 si es perfecto: 1 - Cosine
        cos_sim = F.cosine_similarity(pred_7d, target_7d, dim=-1)
        L_task = torch.mean(1.0 - cos_sim)
        
        # Pérdida Simpléctica (Inercia / Energía Cinética)
        L_symp = 0.5 * torch.mean(pred_7d ** 2)
        
        # Pérdida Métrica (Disipativa hacia la Entropía)
        target_variance = PSI_1
        current_variance = torch.var(pred_7d)
        L_metr = torch.abs(current_variance - target_variance)
        
        total_loss = self.w_vac * L_task + self.w_symp * L_symp + self.w_metr * L_metr
        return total_loss, L_task.item(), L_symp.item(), L_metr.item(), torch.mean(cos_sim).item()

class MetriplecticTrainer:
    def __init__(self, model: nn.Module, device: str = 'cpu', save_dir: str = './working'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.h_symp, self.h_metr, self.h_cos = [], [], []
        self.model.to(device)
    
    def train(self, train_loader: DataLoader, num_epochs: int = 15, lr: float = 0.005):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        criterion = MetriplecticLoss()
        
        logger.info(f"Entrenamiento Tarea Kaggle 128->7 ({num_epochs} Épocas)")
        global_step = 1
        
        for epoch in range(num_epochs):
            self.model.train()
            ep_symp, ep_metr, ep_cos = 0.0, 0.0, 0.0
            
            pbar = tqdm(train_loader, desc=f'Época {epoch+1}/{num_epochs}')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(inputs, step_n=global_step)
                loss, L_task, L_symp, L_metr, cos_sim = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                ep_symp += L_symp
                ep_metr += L_metr
                ep_cos += cos_sim
                global_step += 1
                
                pbar.set_postfix({'Symp': f'{L_symp:.4f}', 'Metr': f'{L_metr:.4f}', 'CosSim': f'{cos_sim:.4f}'})
            
            batches = len(train_loader)
            self.h_symp.append(ep_symp / batches)
            self.h_metr.append(ep_metr / batches)
            self.h_cos.append(ep_cos / batches)
            logger.info(f"Época {epoch+1} | Cosine: {self.h_cos[-1]:.4f} (>0.85 pass) | Symp: {self.h_symp[-1]:.4f} | Metr: {self.h_metr[-1]:.4f}")
            
    def plot_diagnostics(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(self.h_symp, label=f'Simpléctico ($H$) w={W_SYMP:.3f}', color='blue')
        ax1.plot(self.h_metr, label=f'Métrico ($S$) w={W_METR:.3f}', color='red')
        ax1.set_title('Regla 3.3 H7: Competencia Dinámica')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.h_cos, label='Cosine Similarity (Rendimiento Kaggle)', color='green', linewidth=2)
        ax2.axhline(y=0.85, color='orange', linestyle='--', label='Kaggle Pass Threshold (> 0.85)')
        ax2.set_title('The H7 Attention Task - Reconstrucción Holográfica')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, 'metriplectic_rnn_diagnostics.png')
        plt.savefig(path)
        logger.info(f"Gráfico de rendimiento oficial guardado en '{path}'")
        plt.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dev: {device}")
    
    # Simular El Task Oficial
    dataset = HolographicDataset(num_samples=2500) # Entrenamiento robusto ficticio
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = MetriplecticRNN()
    trainer = MetriplecticTrainer(model, device=device)
    
    trainer.train(loader, num_epochs=20, lr=0.005)
    trainer.plot_diagnostics()
