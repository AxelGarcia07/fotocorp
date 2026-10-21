# security.py - Capa de Criptografía y Conexiones Cloud
import boto3
import io
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class AWSManager:
    """
    Gestor de buckets S3 usando Boto3.
    Diseñado bajo el principio de Zero-Storage: Las imágenes se descargan como bytes 
    directo a la memoria RAM (io.BytesIO), nunca tocan el disco duro local,
    optimizando la velocidad de lectura y evitando fuga de datos en el cliente local.
    """
    def __init__(self, credentials_dict):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=credentials_dict.get('aws_access_key_id'),
            aws_secret_access_key=credentials_dict.get('aws_secret_access_key'),
            region_name=credentials_dict.get('region_name', 'us-east-1')
        )
        self.bucket = credentials_dict.get('bucket_name')
        self.prefix = credentials_dict.get('folder_prefix', '')

    def list_images(self):
        """Recupera el árbol de objetos usando paginadores para soportar miles de archivos."""
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
            files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        k = obj['Key']
                        if k.lower().endswith(('.jpg', '.jpeg', '.png')):
                            files.append(k)
            return files
        except Exception as e:
            print(f"AWS Error (List): {e}")
            return []

    def read_image_to_memory(self, key):
        """Descarga el objeto al buffer de memoria volátil."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return io.BytesIO(obj['Body'].read())
        except: return None

    # ... (Resto de funciones copy/delete permanecen igual) ...

def decrypt_data(file_path, password):
    """
    Desencriptación Simétrica de Grado Militar.
    Para evitar almacenar llaves de AWS en texto plano (Hardcoding), se usa un archivo '.rox'.
    Implementa:
    1. PBKDF2HMAC con 480,000 iteraciones (Protección contra fuerza bruta).
    2. Salteado Criptográfico (Salt) de 16 bytes extraído de la cabecera.
    3. Desencriptado final usando Fernet (AES-128 con autenticación HMAC).
    """
    try:
        with open(file_path, 'rb') as f: data = f.read()
        salt = data[:16]
        encrypted_content = data[16:]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(), 
            length=32, 
            salt=salt, 
            iterations=480000
        )
        # Se deriva la llave final a partir del password ingresado por el operador.
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        # Desencriptado y mapeo a diccionario JSON.
        return json.loads(Fernet(key).decrypt(encrypted_content))
    except: 
        return None