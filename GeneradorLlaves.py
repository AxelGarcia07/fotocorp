# GeneradorLlaves.py
# HERRAMIENTA PARA SISTEMAS: Encriptar credenciales para Roxtros
import json
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def generar_llave_maestra(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

print("--- GENERADOR DE LLAVES ROXTROS ---")
json_path = input("Arrastra aquí el archivo .json con las credenciales: ").strip('"')
password = input("Define una contraseña segura para este archivo: ")

# 1. Leer datos reales
with open(json_path, 'rb') as f:
    datos_json = f.read()

# 2. Generar Sal y Llave
salt = os.urandom(16)
key = generar_llave_maestra(password, salt)
f = Fernet(key)

# 3. Encriptar
token = f.encrypt(datos_json)

# 4. Guardar archivo final (Guardamos la Salt + El Token)
nombre_final = "credenciales_roxtros.rox"
with open(nombre_final, 'wb') as out:
    out.write(salt + token)

print(f"\n[ÉXITO] Archivo creado: {nombre_final}")
print("Entrega este archivo y la contraseña al usuario de Roxtros.")