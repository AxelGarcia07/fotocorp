# ai_engine.py - V27.0 (Motor de Inferencia Biométrica)
import cv2
import numpy as np
from PIL import Image
import easyocr
import insightface
from insightface.app import FaceAnalysis
from utils import resource_path
import torch

class FaceRecognitionAI:
    """
    Clase Core para el procesamiento biométrico facial.
    Utiliza un pipeline de 2 etapas: RetinaFace (Detección y Alineación 3D) 
    y ArcFace (Extracción de Características / Embeddings).
    """
    def __init__(self):
        # 1. Detección de Hardware: Delegación de tensores a la GPU (NVIDIA CUDA)
        self.use_gpu = torch.cuda.is_available()
        self.ctx_id = 0 if self.use_gpu else -1
        print(f"AI Engine: Inicializando Inferencia en -> {'GPU (CUDA)' if self.use_gpu else 'CPU'}")

        try:
            # 2. Inicialización del Modelo.
            # NOTA PARA TI: Si 'buffalo_l' no existe localmente, esta función invoca un 
            # request automático para descargar los pesos (.onnx) desde los repositorios de InsightFace.
            self.app = FaceAnalysis(
                name='buffalo_l', 
                allowed_modules=['detection', 'recognition'],
                # Se fuerza el uso de ONNX Runtime con ejecución en C++ y puentes CUDA para máxima velocidad.
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            # Preparamos el tensor de entrada con una resolución máxima de 640x640 para optimizar VRAM.
            self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
            print("✅ IA Facial cargada y montada en VRAM.")
        except Exception as e:
            print(f"❌ Error al inicializar el entorno ONNX: {e}")

    def enhance_face_for_helmet(self, pil_img_crop):
        """
        Filtro de pre-procesamiento adaptativo (CLAHE).
        Convierte el espacio RGB a LAB para ecualizar únicamente la luminiscencia (L).
        Objetivo: Recuperar facciones ocultas bajo sombras duras (ej. cascos o gorras).
        """
        try:
            img_cv = np.array(pil_img_crop.convert('RGB'))
            lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            limg = cv2.merge((l,a,b))
            return Image.fromarray(cv2.cvtColor(limg, cv2.COLOR_LAB2RGB))
        except: 
            return pil_img_crop

    def get_clothing_histogram(self, pil_full_img, face_box):
        """
        Filtro Heurístico Secundario.
        Calcula un histograma de color en el espacio HSV extrayendo el torso del sujeto.
        Objetivo: Penalizar falsos positivos faciales cruzando la telemetría de la ropa.
        """
        try:
            img_cv = np.array(pil_full_img.convert('RGB'))
            h, w, _ = img_cv.shape
            
            # Coordenadas Bounding Box de la cara
            x1, y1, x2, y2 = [int(c) for c in face_box]
            face_h = y2 - y1
            face_w = x2 - x1
            
            # Estimación anatómica del torso (1.5x el tamaño de la cara hacia abajo)
            torso_y1 = y2 
            torso_y2 = min(h, y2 + int(face_h * 1.5)) 
            torso_x1 = max(0, x1 - int(face_w * 0.2))
            torso_x2 = min(w, x2 + int(face_w * 0.2))
            
            if torso_y1 >= torso_y2 or torso_x1 >= torso_x2:
                return None 

            # Recorte y conversión a HSV (Tono, Saturación, Valor) para ignorar cambios de luz.
            torso_crop = img_cv[torso_y1:torso_y2, torso_x1:torso_x2]
            hsv_crop = cv2.cvtColor(torso_crop, cv2.COLOR_RGB2HSV)
            
            # Generación de la firma de color (Histograma)
            hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist
        except: 
            return None

    def compare_clothing(self, hist1, hist2):
        """Calcula la correlación entre dos histogramas. (1.0 = idénticos, 0.0 = distintos)"""
        if hist1 is None or hist2 is None: return 0.5 
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def get_faces(self, pil_img):
        """
        Ejecuta la inferencia sobre la imagen completa.
        Retorna: Arrays con los vectores de 512-dim (ADN Facial) y las cajas delimitadoras.
        """
        if pil_img is None: return [], []
        try:
            # InsightFace requiere formato BGR (estándar de OpenCV)
            img_bgr = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
            faces = self.app.get(img_bgr)
            
            embeddings = []
            valid_boxes = []
            
            for face in faces:
                embeddings.append(face.embedding.flatten())
                valid_boxes.append(face.bbox.astype(int).tolist())

            return embeddings, valid_boxes
        except Exception as e: 
            print(f"Error Inferencia Facial: {e}")
            return [], []

    def get_single_embedding(self, pil_img_crop):
        """Inferencia para un recorte facial específico (Modo búsqueda)."""
        if pil_img_crop is None: return None
        try:
            img_bgr = cv2.cvtColor(np.array(pil_img_crop.convert('RGB')), cv2.COLOR_RGB2BGR)
            faces = self.app.get(img_bgr)
            if faces:
                return faces[0].embedding.reshape(1, -1)
            return None
        except: return None

class NumberRecognitionAI:
    """Microservicio OCR para extraer telemetría numérica (Dorsales/Brazaletes)"""
    def __init__(self):
        try: 
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), download_enabled=False)
        except: 
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), download_enabled=True)
            
    def get_numbers(self, pil):
        try:
            img = np.array(pil.convert('RGB'))
            # Se restringe el charset para evitar falsos positivos con letras (solo números)
            res = self.reader.readtext(img, allowlist='0123456789')
            return [(t, p) for (b, t, p) in res if p > 0.4 and len(t)>=2]
        except: 
            return []