# faiss_engine.py - Motor de Búsqueda Vectorial por Similitud
import numpy as np
import faiss
import torch

class FaissSearchEngine:
    """
    Implementación del índice vectorial FAISS (Meta AI).
    Sustituye el recorrido secuencial O(N^2) por un Indexado Espacial.
    """
    def __init__(self, dimension=512):
        self.dimension = dimension
        self.use_gpu = torch.cuda.is_available()
        
        # 1. Crear índice CPU base (Inner Product).
        # Nota: Al normalizar los vectores mediante L2, el Inner Product se convierte
        # matemáticamente en Similitud Coseno (Cosine Similarity).
        self.index_cpu = faiss.IndexFlatIP(self.dimension)
        
        # 2. Migración del índice a la VRAM de la GPU para búsqueda paralela masiva.
        if self.use_gpu:
            try:
                self.res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index_cpu)
                print("⚡ FAISS Engine: Aceleración por hardware (CUDA) activada.")
            except Exception as e:
                print(f"⚠️ FAISS fallback a CPU ({e}).")
                self.index = self.index_cpu
        else:
            self.index = self.index_cpu
            print("💻 FAISS Engine: Instanciado en CPU.")
            
        self.paths_map = [] # Diccionario que vincula el Index en memoria con la ruta del archivo.

    def reset(self):
        """Flushea el índice en memoria para evitar colisiones en un re-scan."""
        self.index.reset()
        self.paths_map = []

    def build_index(self, embeddings_list, paths_list):
        """
        Ingesta de Datos masiva (Data Ingestion).
        Recibe las matrices de vectores, las normaliza en L2 y las carga a la RAM/VRAM.
        """
        if not embeddings_list: return

        self.reset()
        
        # Conversión a tensores de punto flotante de 32-bits (Estándar de IA)
        data = np.array(embeddings_list, dtype=np.float32)
        faiss.normalize_L2(data)
        
        self.index.add(data)
        self.paths_map = list(paths_list)
        print(f"✅ FAISS: {self.index.ntotal} vectores mapeados en el espacio métrico.")

    def search_target(self, target_embedding, k=50, threshold_distance=0.40):
        """
        Realiza un Query K-NN (K-Nearest Neighbors).
        Busca las K coincidencias más cercanas al vector objetivo.
        """
        if self.index.ntotal == 0: return []

        query = np.array([target_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        # Retorna matrices con los "scores" (similitud) y los índices de las fotos.
        similarities, indices = self.index.search(query, min(k, self.index.ntotal))

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1: continue
            
            # Conversión inversa: Distancia = 1 - Similitud. (Distancia 0.0 es idéntico)
            cosine_dist = max(0.0, 1.0 - float(sim))
            
            if cosine_dist <= threshold_distance:
                path = self.paths_map[idx]
                results.append((path, cosine_dist))

        return results