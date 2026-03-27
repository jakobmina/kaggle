import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- H7 Constantes ---
PHI = (1 + math.sqrt(5)) / 2
PSI_1 = abs(math.cos(math.pi * PHI))
DRIFT_072 = 7 - 2 * math.pi  # ~ 0.7168
# Pesos propuestos (71 - 28 - 1)
W_SYMP = DRIFT_072       # ~ 0.7168 Conservación/Inercia (Base)
W_METR = 1.0 - DRIFT_072 - 0.01 # ~ 0.2732 Disipación/Entropía
W_VAC  = 0.01            # Término de vacío / Residuos

class GoldenModulationLayer(nn.Module):
    """
    Inyecta el Operador Áureo a la capa oculta.
    O_n = cos(pi * n) * cos(pi * phi * n)
    """
    def __init__(self):
        super(GoldenModulationLayer, self).__init__()
        
    def forward(self, x: torch.Tensor, step_n: int) -> torch.Tensor:
        # Evaluamos el operador áureo de manera escalar para el paso 'n'
        O_n = math.cos(math.pi * step_n) * math.cos(math.pi * PHI * step_n)
        # Modulamos el tensor 
        return x * O_n

class TernaryDataset(Dataset):
    """
    Dataset adaptado a firmas continuas/ternarias del benchmark H7.
    Convierte trazas de estados biológicos/informacionales [-1, 0, 1] 
    a índices continuos o categóricos [0, 1, 2] protegidos.
    """
    def __init__(self, sequences_h7: List[List[float]], max_length: int = 128, stride: int = 10):
        self.max_length = max_length
        self.stride = stride
        self.sequences = []
        self._prepare_sequences(sequences_h7)
        
    def _prepare_sequences(self, sequences: List[List[float]]) -> None:
        logger.info("Preparando secuencias ternarias/cuánticas...")
        for seq in sequences:
            # Ventanas deslizantes a lo largo del tract cognitivo
            for i in range(0, len(seq) - 1, self.stride):
                window = seq[i:i + self.max_length]
                if len(window) > 1:
                    self.sequences.append(window)
        logger.info(f"Creadas {len(self.sequences)} secuencias H7 de entrenamiento")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        # Para secuencias temporales H7: entrada (t), objetivo a predecir (t+1)
        # +1 para desplazar el ternario cerrado de [-1, 0, 1] a índices [0, 1, 2]
        input_ids = torch.tensor([s + 1 for s in seq[:-1]], dtype=torch.long)
        target_ids = torch.tensor([s + 1 for s in seq[1:]], dtype=torch.long)
        return input_ids, target_ids

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=3)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)
    return inputs_padded, targets_padded

class MetriplecticRNN(nn.Module):
    """
    Procesador Metripléctico Multicapa (RNN Base + Modulación H7).
    """
    def __init__(self, vocab_size: int = 4, embed_size: int = 64, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(MetriplecticRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Proyección de estados discretos a dimensión densa
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=3)
        
        # Memoria/Inercia (Simpléctica equivalente aproximada)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Modulación Áurea O_n
        self.golden_modulation = GoldenModulationLayer()
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Salida colapsando de nuevo al estado ternario
        self.fc = nn.Linear(hidden_size, vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param) if param.dim() > 1 else nn.init.normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, step_n: int, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(x)  
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Inyectando Vacío Estructurado / Modulación H7 al hidden state general
        modulated_out = self.golden_modulation(rnn_out, step_n)
        
        norm_out = self.layer_norm(modulated_out)
        norm_out = self.dropout(norm_out)
        
        output = self.fc(norm_out)  
        return output, hidden

class MetriplecticLoss(nn.Module):
    """
    Función de Pérdida Dual Regida por Constantes H7 (71.7% - 28.3% - 1%)
    Valida la Regla 1.1 y 1.2 del Mandato Metripléctico.
    """
    def __init__(self, w_symp=W_SYMP, w_metr=W_METR, w_vac=W_VAC):
        super(MetriplecticLoss, self).__init__()
        self.task_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.w_symp = w_symp
        self.w_metr = w_metr
        self.w_vac = w_vac
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        # Tareas: Token Prediction Error
        L_task = self.task_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Pérdida Simpléctica (Energía Cinetica equivalente: H = 1/2 psi^2)
        # Queremos conservar inercia pero sin explotar. Penalizamos magnitudes exabruptas de estado.
        L_symp = 0.5 * torch.mean(logits ** 2)
        
        # Pérdida Métrica (Disipativa/Relajación)
        # Simulando la entropía S ln(psi): empuja las distribuciones hacia un equilibrio.
        # Aproximamos penalizando la varianza general (que no se quede atrapado ni fluctúe a infinito)
        target_variance = PSI_1  # Relaja hacia la entropía áurea
        current_variance = torch.var(logits)
        L_metr = torch.abs(current_variance - target_variance)
        
        # Ecuación Total: L_total = W_vac * L_task + W_symp * L_symp + W_metr * L_metr
        # Usaremos L_task como el término de vacío 'W_vac' estructural guiado por los datos experimentales
        total_loss = self.w_vac * L_task + self.w_symp * L_symp + self.w_metr * L_metr
        
        return total_loss, L_symp.item(), L_metr.item()

class MetriplecticTrainer:
    """Clase para manejar el entrenamiento y rastrear diagnósticos H7"""
    def __init__(self, model: nn.Module, device: str = 'cpu', save_dir: str = './checkpoints'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.history_symp = []
        self.history_metr = []
        self.history_total = []
        self.model.to(device)
    
    def train(self, train_loader: DataLoader, num_epochs: int = 10, lr: float = 0.001):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = MetriplecticLoss()
        
        logger.info(f"Iniciando Entrenamiento Metripléctico ({num_epochs} Épocas)")
        
        global_step = 1 # Step n para el Golden Operator
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss, train_symp, train_metr = 0.0, 0.0, 0.0
            
            progress_bar = tqdm(train_loader, desc=f'Época {epoch+1}/{num_epochs}')
            
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                outputs, _ = self.model(inputs, step_n=global_step)
                loss, L_symp, L_metr = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_symp += L_symp
                train_metr += L_metr
                global_step += 1
                
                progress_bar.set_postfix({'Symp(H)': f'{L_symp:.4f}', 'Metr(S)': f'{L_metr:.4f}'})
            
            self.history_total.append(train_loss / len(train_loader))
            self.history_symp.append(train_symp / len(train_loader))
            self.history_metr.append(train_metr / len(train_loader))
            
            logger.info(f"Época {epoch+1} | Loss Total: {self.history_total[-1]:.4f} | Symp: {self.history_symp[-1]:.4f} | Metr: {self.history_metr[-1]:.4f}")
            
    def plot_diagnostics(self):
        """Regla 3.3 Visualización Diagnóstica - Competencia H vs S"""
        plt.figure(figsize=(12, 5))
        
        # Grafica la competencia Metripléctica
        plt.plot(self.history_symp, label=f'Inercial/Simpléctico ($H$) w={W_SYMP:.3f}', color='blue', alpha=0.8)
        plt.plot(self.history_metr, label=f'Relajación/Métrico ($S$) w={W_METR:.3f}', color='red', alpha=0.8)
        plt.title('Competencia Metripléctica en la Función de Pérdida ($H$ vs $S$)')
        plt.xlabel('Época')
        plt.ylabel('Magnitud de Componente')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, 'metriplectic_rnn_diagnostics.png')
        plt.savefig(path)
        logger.info(f"Diagnóstico guardado en '{path}'")
        plt.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")
    
    # 1. Simular Firmware Cognitivo H7 (Series discretas de estados ternarios [-1, 0, 1])
    np.random.seed(42)
    # Genera 100 secuencias sintéticas oscilatorias entre -1, 0 y 1 para prueba
    sample_h7_tracks = [np.round(np.sin(np.linspace(0, 4*np.pi, 50)) * np.random.choice([0.5, 1.0])).astype(int).tolist() for _ in range(100)]
    
    # 2. Dataset y Dataloader Ternario
    dataset = TernaryDataset(sample_h7_tracks, max_length=15, stride=2)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # 3. Inicializar Modelo
    model = MetriplecticRNN(vocab_size=4, embed_size=32, hidden_size=64, num_layers=1)
    
    # 4. Entrenar y Diagnosticar
    trainer = MetriplecticTrainer(model, device=device, save_dir='./working')
    trainer.train(loader, num_epochs=20, lr=0.005)
    trainer.plot_diagnostics()
