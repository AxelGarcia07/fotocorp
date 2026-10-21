import pyodbc

class ParkDBManager:
    def __init__(self, creds=None):
        self.connected = False
        self.conn_str = ""
        self.table = ""
        self.col_tag = ""
        self.col_path = ""
        
        if creds:
            self.configure(creds)

    def configure(self, creds):
        """Recibe el diccionario desencriptado y configura la conexión"""
        try:
            server = creds.get('db_server')
            database = creds.get('db_name')
            username = creds.get('db_user')
            password = creds.get('db_pass')
            driver = creds.get('db_driver', '{ODBC Driver 17 for SQL Server}')
            
            # Configuración de columnas para la consulta dinámica
            self.table = creds.get('db_table', 'fotos')
            self.col_tag = creds.get('col_tag', 'tag_id')
            self.col_path = creds.get('col_path', 'path')

            self.conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
            self.connected = True
            print("DB Manager: Credenciales configuradas.")
        except Exception as e:
            print(f"Error configurando DB: {e}")
            self.connected = False

    def search_tag(self, tag_id):
        if not self.connected:
            print("DB no configurada. Carga la Llave Maestra primero.")
            return []
            
        try:
            # --- CONEXIÓN REAL ---
            # conn = pyodbc.connect(self.conn_str)
            # cursor = conn.cursor()
            # query = f"SELECT {self.col_path} FROM {self.table} WHERE {self.col_tag} = ?"
            # cursor.execute(query, tag_id)
            # rows = cursor.fetchall()
            # return [row[0] for row in rows]
            
            # --- MODO SIMULACIÓN (Para que veas que funciona la lógica) ---
            print(f"Consultando SQL: SELECT {self.col_path} FROM {self.table} WHERE {self.col_tag} = '{tag_id}'")
            return [] 

        except Exception as e:
            print(f"Error SQL: {e}")
            return []