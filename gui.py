import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import threading
import time
import shutil
import json
import numpy as np
import faiss 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
import torch
import pickle 
import logging 

# Importamos nuestros módulos propios
from utils import resource_path
from security import AWSManager, decrypt_data
from ai_engine import FaceRecognitionAI, NumberRecognitionAI
from faiss_engine import FaissSearchEngine
from park_db import ParkDBManager

# --- CONFIGURACIÓN DE LA BITÁCORA (LOGS) ---
logging.basicConfig(
    filename='roxtros_system.log',
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info("=========================================")
logging.info("🚀 SISTEMA ROXTROS INICIADO - V24.0 (Armor, Logs & FAISS Graph)")
logging.info("=========================================")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# ==========================================
#   COMPONENTES VISUALES
# ==========================================
class SplashScreen(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        w, h = 500, 350
        x = (self.winfo_screenwidth()/2) - (w/2); y = (self.winfo_screenheight()/2) - (h/2)
        self.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.configure(fg_color="#101010")
        try:
            img = Image.open(resource_path("roxtros_logo.png"))
            self.logo = ctk.CTkImage(img, size=(130, 130))
            ctk.CTkLabel(self, image=self.logo, text="").pack(pady=40)
        except: ctk.CTkLabel(self, text="🦜", font=("Arial", 100)).pack(pady=40)
        ctk.CTkLabel(self, text="ROXTROS", font=("Impact", 42), text_color="#2CC985").pack()
        ctk.CTkLabel(self, text="Enterprise v24.0 (Armor, Logs & FAISS Graph)", font=("Arial", 12), text_color="gray").pack()
        self.progress = ctk.CTkProgressBar(self, width=350, progress_color="#2CC985"); self.progress.pack(pady=20)
        self.after(50, self.update_loading); self.step = 0
    def update_loading(self):
        self.step += 1; self.progress.set(self.step / 30)
        if self.step < 30: self.after(30, self.update_loading)
        else: self.destroy(); RoxtrosApp().mainloop()

class ImageViewer(ctk.CTkToplevel):
    def __init__(self, app, paths, start_idx, face_ai_engine=None):
        super().__init__(app)
        self.app = app
        self.paths = paths
        self.current_idx = start_idx
        self.face_ai_engine = face_ai_engine
        
        self.geometry("1000x850")
        self.transient(app) 
        self.after(10, self.lift)
        self.focus_force()
        
        self.ctrl_frame = ctk.CTkFrame(self, height=50)
        self.ctrl_frame.pack(fill="x", side="top", pady=5, padx=10)
        
        ctk.CTkButton(self.ctrl_frame, text="<< Anterior", command=self.prev_img, width=100).pack(side="left", padx=10)
        self.lbl_title = ctk.CTkLabel(self.ctrl_frame, text="", font=("Arial", 14, "bold"))
        self.lbl_title.pack(side="left", expand=True)
        ctk.CTkButton(self.ctrl_frame, text="Siguiente >>", command=self.next_img, width=100).pack(side="right", padx=10)
        
        self.bind("<Left>", lambda e: self.prev_img())
        self.bind("<Right>", lambda e: self.next_img())
        
        self.canvas = tk.Canvas(self, bg="#101010", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.imscale = 1.0; self.delta = 1.3
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>', self.move_to)
        
        self._resize_job = None
        self.canvas.bind('<Configure>', self.on_canvas_resize)
        
        self.load_current()

    def on_canvas_resize(self, event):
        if event.widget == self.canvas:
            if self._resize_job:
                self.after_cancel(self._resize_job)
            self._resize_job = self.after(100, self.fit_to_window)

    def load_current(self):
        path = self.paths[self.current_idx]
        self.title(f"Visor: {os.path.basename(path)}")
        self.lbl_title.configure(text=f"Foto {self.current_idx + 1} de {len(self.paths)} - {os.path.basename(path)}")
        
        pil_image_obj = self.app.load_image_unified(path)
        if not pil_image_obj: return
        self.pil_base = pil_image_obj.convert('RGB')
        
        try: font = ImageFont.truetype("arial.ttf", 40)
        except: font = None

        try:
            if self.face_ai_engine:
                _, boxes = self.face_ai_engine.get_faces(self.pil_base)
                if boxes is not None:
                    draw = ImageDraw.Draw(self.pil_base)
                    
                    if self.app.mode_var.get() == "Buscar: Por Foto" and self.app.ref_target_emb is not None:
                        for box in boxes:
                            b = [int(x) for x in box]
                            face_crop = self.pil_base.crop((b[0], b[1], b[2], b[3]))
                            curr_emb = self.face_ai_engine.get_single_embedding(face_crop)
                            curr_hist = self.face_ai_engine.get_clothing_histogram(self.pil_base, b)
                            
                            if curr_emb is not None:
                                dist = cosine_distances(self.app.ref_target_emb, curr_emb)[0][0]
                                if not self.app.chk_clothes_var.get() and self.app.ref_target_hist is not None and curr_hist is not None:
                                    similarity = self.face_ai_engine.compare_clothing(self.app.ref_target_hist, curr_hist)
                                    if similarity < 0.2: dist += 0.3 
                                    elif similarity > 0.6: dist -= 0.1
                                
                                color = "#FF0000" 
                                width = 3
                                if dist < self.app.slider.get(): 
                                    color = "#00FF00" 
                                    width = 6
                                
                                draw.rectangle(b, outline=color, width=width)
                                draw.text((b[0], b[1]-40), f"{dist:.2f}", fill=color, font=font)
                    else:
                        for box in boxes:
                            draw.rectangle(box, outline="#3498DB", width=3)
        except: pass
        
        self.original_image = self.pil_base.copy()
        self.tk_img = None
        self.after(50, self.fit_to_window) 

    def prev_img(self):
        if self.current_idx > 0: self.current_idx -= 1; self.load_current()
    def next_img(self):
        if self.current_idx < len(self.paths) - 1: self.current_idx += 1; self.load_current()
        
    def fit_to_window(self):
        w_canvas = self.canvas.winfo_width(); h_canvas = self.canvas.winfo_height()
        w_img, h_img = self.original_image.size
        if w_canvas > 1 and h_canvas > 1: 
            self.imscale = min(w_canvas/w_img, h_canvas/h_img)
            self.show_image()
            
    def show_image(self):
        if self.imscale <= 0: self.imscale = 0.1
        w, h = self.original_image.size
        resized = self.original_image.resize((int(w*self.imscale), int(h*self.imscale)), Image.Resampling.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, anchor='center', image=self.tk_img)
        
    def wheel(self, e):
        self.imscale *= self.delta if e.delta > 0 else 1/self.delta; self.show_image()
    def move_from(self, e): self.canvas.scan_mark(e.x, e.y)
    def move_to(self, e): self.canvas.scan_dragto(e.x, e.y, gain=1)

class ComparisonViewer(ctk.CTkToplevel):
    def __init__(self, app, ref_path, cand_paths, start_idx, threshold):
        super().__init__(app)
        self.app = app
        self.ref_path = ref_path
        self.cand_paths = cand_paths
        self.current_idx = start_idx
        self.threshold = threshold 
        
        self.title(f"Comparación (Umbral: {threshold})")
        self.geometry("1200x800")
        self.transient(app)
        self.after(10, self.lift)
        self.focus_force()

        self.ctrl_frame = ctk.CTkFrame(self, fg_color="transparent", height=40)
        self.ctrl_frame.pack(fill="x", pady=5)
        ctk.CTkButton(self.ctrl_frame, text="<< Anterior", command=self.prev_img, width=120).pack(side="left", padx=20)
        self.lbl_counter = ctk.CTkLabel(self.ctrl_frame, text="", font=("Arial", 14, "bold"))
        self.lbl_counter.pack(side="left", expand=True)
        ctk.CTkButton(self.ctrl_frame, text="Siguiente >>", command=self.next_img, width=120).pack(side="right", padx=20)

        self.bind("<Left>", lambda e: self.prev_img())
        self.bind("<Right>", lambda e: self.next_img())

        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.grid_columnconfigure(0, weight=1); self.main_frame.grid_columnconfigure(1, weight=1); self.main_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.main_frame, text="REFERENCIA (Tu Input)", font=("Arial", 16, "bold"), text_color="orange").grid(row=0, column=0, pady=5)
        self.lbl_cand_title = ctk.CTkLabel(self.main_frame, text="CANDIDATO", font=("Arial", 16, "bold"), text_color="#2CC985")
        self.lbl_cand_title.grid(row=0, column=1, pady=5)

        self.target_w, self.target_h = 580, 650
        
        pil_ref = self.app.load_image_unified(self.ref_path)
        self.ref_base_pil = pil_ref.copy() if pil_ref else Image.new('RGB', (100,100))
        
        try:
            if self.app.ref_target_box is not None:
                d_ref = ImageDraw.Draw(self.ref_base_pil)
                d_ref.rectangle(self.app.ref_target_box, outline="yellow", width=6)
        except: pass

        ref_thumb = self.ref_base_pil.copy()
        ref_thumb.thumbnail((self.target_w, self.target_h), Image.Resampling.LANCZOS)
        self.tk_ref = ctk.CTkImage(ref_thumb, size=ref_thumb.size)
        
        self.lbl_img_ref = ctk.CTkLabel(self.main_frame, image=self.tk_ref, text="")
        self.lbl_img_ref.grid(row=1, column=0, padx=5, pady=5)
        
        self.lbl_img_cand = ctk.CTkLabel(self.main_frame, text="Cargando...")
        self.lbl_img_cand.grid(row=1, column=1, padx=5, pady=5)

        self.load_current()

    def load_current(self):
        cand_path = self.cand_paths[self.current_idx]
        self.lbl_counter.configure(text=f"Foto {self.current_idx + 1} de {len(self.cand_paths)}")
        self.lbl_cand_title.configure(text=f"CANDIDATO: {os.path.basename(cand_path)}")
        
        pil_cand = self.app.load_image_unified(cand_path)
        if pil_cand:
            img_cand_pil = pil_cand.copy().convert("RGB")
            self.draw_target_boxes(img_cand_pil) 
            img_cand_pil.thumbnail((self.target_w, self.target_h), Image.Resampling.LANCZOS)
            self.tk_cand = ctk.CTkImage(img_cand_pil, size=img_cand_pil.size)
            self.lbl_img_cand.configure(image=self.tk_cand, text="")
        else:
            self.lbl_img_cand.configure(image="", text="Error al leer foto")

    def prev_img(self):
        if self.current_idx > 0: self.current_idx -= 1; self.load_current()
    def next_img(self):
        if self.current_idx < len(self.cand_paths) - 1: self.current_idx += 1; self.load_current()

    def draw_target_boxes(self, img_cand):
        if not self.app.face_ai or self.app.ref_target_emb is None: return
        draw = ImageDraw.Draw(img_cand)
        
        try: font = ImageFont.truetype("arial.ttf", 40)
        except: font = None

        target_emb = self.app.ref_target_emb
        ref_hist = self.app.ref_target_hist

        try:
            _, boxes = self.app.face_ai.get_faces(img_cand)
            if not boxes: return
            
            for box in boxes:
                b = [int(x) for x in box]
                face_crop = img_cand.crop((b[0], b[1], b[2], b[3]))
                curr_emb = self.app.face_ai.get_single_embedding(face_crop)
                curr_hist = self.app.face_ai.get_clothing_histogram(img_cand, b)

                if curr_emb is not None:
                    dist = cosine_distances(target_emb, curr_emb)[0][0]
                    if not self.app.chk_clothes_var.get() and ref_hist is not None and curr_hist is not None:
                        similarity = self.app.face_ai.compare_clothing(ref_hist, curr_hist)
                        if similarity < 0.2: dist += 0.3 
                        elif similarity > 0.6: dist -= 0.1
                    
                    color = "#FF0000" 
                    width = 3
                    if dist < self.threshold: 
                        color = "#00FF00" 
                        width = 8
                    
                    draw.rectangle(b, outline=color, width=width)
                    text = f"{dist:.2f}"
                    draw.text((b[0], b[1]-40), text, fill=color, font=font)
        except: pass

class SmartMoveDialog(ctk.CTkToplevel):
    def __init__(self, parent, existing_options):
        super().__init__(parent)
        self.geometry("400x200")
        self.transient(parent)
        self.after(10, self.lift)
        self.grab_set() 
        self.user_choice = None
        ctk.CTkLabel(self, text="Destino:").pack(pady=5)
        self.combo = ctk.CTkComboBox(self, values=existing_options, width=300); self.combo.pack(pady=10)
        ctk.CTkButton(self, text="Mover", command=self.on_ok, fg_color="#2CC985").pack(pady=10)
    def on_ok(self): self.user_choice = self.combo.get(); self.destroy()
    def get_input(self): self.master.wait_window(self); return self.user_choice

class GalleryPopup(ctk.CTkToplevel):
    def __init__(self, app, paths, key):
        super().__init__(app)
        self.app=app; self.all_paths=paths; self.group_key=key; self.current_page=0; self.items_per_page=24
        self.title(f"Grupo: {key} ({len(paths)} fotos)")
        self.geometry("1100x850")
        self.transient(app)
        self.after(10, self.lift)
        self.focus_force()
        
        self.top=ctk.CTkFrame(self); self.top.pack(fill="x", padx=10, pady=5)
        self.left_panel = ctk.CTkFrame(self.top, fg_color="transparent"); self.left_panel.pack(side="left")
        
        if self.app.mode_var.get() == "Buscar: Por Foto" and self.app.ref_photo_path:
            try:
                pil_ref = self.app.load_image_unified(self.app.ref_photo_path)
                if pil_ref:
                    pil_ref.thumbnail((80, 80))
                    self.img_ref = ctk.CTkImage(pil_ref, size=pil_ref.size)
                    ctk.CTkLabel(self.left_panel, text="Ref:", font=("Arial", 10, "bold"), text_color="orange").pack(side="left")
                    ctk.CTkLabel(self.left_panel, image=self.img_ref, text="").pack(side="left", padx=5)
            except: pass
        
        self.lbl_status = ctk.CTkLabel(self.left_panel, text="Analizado", text_color="gray"); self.lbl_status.pack(side="left", padx=10)
        self.right_panel = ctk.CTkFrame(self.top, fg_color="transparent"); self.right_panel.pack(side="right")
        ctk.CTkButton(self.right_panel, text="<<", width=50, command=self.prev).pack(side="left")
        self.lbl_page = ctk.CTkLabel(self.right_panel, text="Pág 1"); self.lbl_page.pack(side="left")
        ctk.CTkButton(self.right_panel, text=">>", width=50, command=self.next).pack(side="left")
        self.scroll=ctk.CTkScrollableFrame(self); self.scroll.pack(fill="both", expand=True)
        self.render_page()

    def render_page(self):
        for w in self.scroll.winfo_children(): w.destroy()
        start = self.current_page * self.items_per_page
        chunk = self.all_paths[start : start+self.items_per_page]
        self.lbl_page.configure(text=f"Pág {self.current_page+1}")
        for i, p in enumerate(chunk): threading.Thread(target=self.load_card, args=(i, p)).start() 

    def load_card(self, i, p):
        try:
            pil = self.app.load_image_unified(p)
            if pil:
                pil.thumbnail((150,150))
                self.after(0, lambda: self.draw_card(i, p, pil))
        except: pass

    def draw_card(self, i, p, pil):
        f=ctk.CTkFrame(self.scroll); f.grid(row=i//4, column=i%4, padx=5, pady=5)
        ci=ctk.CTkImage(pil, size=pil.size); lbl=ctk.CTkLabel(f, image=ci, text="", cursor="hand2"); lbl.pack()
        
        idx = self.all_paths.index(p)
        lbl.bind("<Button-1>", lambda e, current_idx=idx: ImageViewer(self.app, self.all_paths, current_idx, self.app.face_ai))
        
        btn_frm = ctk.CTkFrame(f, fg_color="transparent"); btn_frm.pack(pady=2)
        
        ctk.CTkButton(btn_frm, text="Mover", width=50, command=lambda path=p: self.req_move(path)).pack(side="left", padx=1)
        if self.app.mode_var.get() == "Buscar: Por Foto":
             ctk.CTkButton(btn_frm, text="Comp", width=50, fg_color="#3498DB", command=lambda path=p: self.launch_comparison(path)).pack(side="left", padx=1)
        if self.app.mode_var.get() in ["Agrupar: Rostros", "Buscar: Por Foto"]:
            ctk.CTkButton(btn_frm, text="Atraer", width=50, fg_color="purple", command=lambda path=p: self.start_magnet_thread(path)).pack(side="left", padx=1)

    def launch_comparison(self, cand_path):
        if not self.app.ref_photo_path: 
            messagebox.showerror("Error", "No hay foto de referencia."); return
        try: start_idx = self.all_paths.index(cand_path)
        except ValueError: start_idx = 0
        ComparisonViewer(self.app, self.app.ref_photo_path, self.all_paths, start_idx, self.app.slider.get()) 

    def next(self): self.current_page+=1; self.render_page()
    def prev(self): 
        if self.current_page>0: self.current_page-=1; self.render_page()
    def req_move(self, p):
        d=SmartMoveDialog(self, self.app.get_existing_group_names()).get_input()
        if d and self.app.move_photo_logic(p, self.group_key, d): 
            if p in self.all_paths: self.all_paths.remove(p); self.render_page()

    def start_magnet_thread(self, p):
        if not messagebox.askyesno("Confirmar Imán", "¿Buscar coincidencias de esta foto en TODA la caché?\n¡Será rapidísimo!"): return
        
        current_mode = self.app.mode_var.get()
        current_thresh = self.app.slider.get()
        
        self.lbl_status.configure(text="Imán activo...", text_color="orange")
        logging.info(f"🟢 IMÁN INICIADO: Buscando coincidencias para la foto: {p}")
        threading.Thread(target=self.run_magnet_worker, args=(p, current_mode, current_thresh), daemon=True).start()

    def run_magnet_worker(self, mag_path, mode, thresh):
        try:
            if not self.app.face_ai: return
            if not getattr(self.app, 'faiss_engine', None): self.app.faiss_engine = FaissSearchEngine()
            
            mag_data = self.app.ai_memory_cache.get(mag_path)
            
            if mag_data and mag_data.get('embs'):
                l_idx = 0; max_area = 0
                for idx, b in enumerate(mag_data['boxes']):
                    area = (b[2]-b[0]) * (b[3]-b[1])
                    if area > max_area: max_area = area; l_idx = idx
                
                target = mag_data['embs'][l_idx].reshape(1, -1)
                mag_hist = mag_data['hists'][l_idx] if 'hists' in mag_data else None
            else:
                pil = self.app.load_image_unified(mag_path)
                if not pil: return
                mag_embs, mag_boxes = self.app.face_ai.get_faces(pil)
                if not mag_embs: 
                    self.app.after(0, lambda: messagebox.showerror("Error", "No detecto cara clara en la foto imán."))
                    self.app.after(0, lambda: self.lbl_status.configure(text="Analizado", text_color="gray"))
                    return
                l_idx = 0; max_area = 0
                for idx, b in enumerate(mag_boxes):
                    area = (b[2]-b[0]) * (b[3]-b[1])
                    if area > max_area: max_area = area; l_idx = idx
                target = mag_embs[l_idx].reshape(1, -1)
                mag_hist = self.app.face_ai.get_clothing_histogram(pil, mag_boxes[l_idx])

            count = 0
            candidates = []
            if mode == "Buscar: Por Foto":
                candidates = [x for x in self.app.processed_files if x not in self.all_paths]
            else:
                for grp in list(self.app.clusters_found.keys()):
                    if grp == self.group_key: continue
                    candidates.extend(self.app.clusters_found[grp]['paths'])

            total_cands = len(candidates)
            
            for i, p in enumerate(candidates):
                if i % 100 == 0: 
                    self.app.after(0, lambda curr=i, tot=total_cands: self.lbl_status.configure(text=f"Imán extrayendo: {curr}/{tot}...", text_color="orange"))
                if p not in self.app.ai_memory_cache:
                    cand_pil = self.app.load_image_unified(p)
                    if cand_pil:
                        curr_embs, curr_boxes = self.app.face_ai.get_faces(cand_pil)
                        if curr_embs:
                            hists = [self.app.face_ai.get_clothing_histogram(cand_pil, b) for b in curr_boxes]
                            self.app.ai_memory_cache[p] = {'embs': curr_embs, 'boxes': curr_boxes, 'hists': hists}
                            
            self.app.after(0, lambda: self.lbl_status.configure(text="Generando Index FAISS...", text_color="orange"))
            emb_list = []
            cand_paths = []
            for p in candidates:
                if p in self.app.ai_memory_cache and self.app.ai_memory_cache[p].get('embs'):
                    for idx, emb in enumerate(self.app.ai_memory_cache[p]['embs']):
                        emb_list.append(emb)
                        cand_paths.append((p, idx))
                        
            matched_paths = set()
            if emb_list:
                self.app.faiss_engine.build_index(emb_list, cand_paths)
                faiss_results = self.app.faiss_engine.search_target(target.flatten(), k=len(emb_list), threshold_distance=thresh + 0.15)
                
                for (p, face_idx), dist in faiss_results:
                    if dist < 0.6 and not self.app.chk_clothes_var.get() and mag_hist is not None:
                        curr_hist = self.app.ai_memory_cache[p]['hists'][face_idx] if p in self.app.ai_memory_cache and 'hists' in self.app.ai_memory_cache[p] else None
                        if curr_hist is not None:
                            sim = self.app.face_ai.compare_clothing(mag_hist, curr_hist)
                            if sim < 0.2: dist += 0.3
                            elif sim > 0.6: dist -= 0.1
                    if dist < thresh:
                        matched_paths.add(p)

            for p in matched_paths:
                if mode == "Buscar: Por Foto":
                    if p not in self.all_paths: 
                        self.all_paths.append(p); count += 1
                        if "Coincidencias" in self.app.clusters_found:
                            self.app.clusters_found["Coincidencias"]['paths'].append(p)

            logging.info(f"✅ IMÁN COMPLETADO: {count} fotos atraídas exitosamente.")
            self.app.after(0, lambda: self.magnet_done(count))
            
        except Exception as e:
            logging.error(f"❌ ERROR EN IMÁN: {e}", exc_info=True)
            self.app.after(0, lambda err=e: messagebox.showerror("Error Imán", str(err)))
            self.app.after(0, lambda: self.lbl_status.configure(text="Analizado", text_color="gray"))

    def magnet_done(self, count):
        self.lbl_status.configure(text="Analizado", text_color="gray")
        messagebox.showinfo("Imán Completado", f"Operación finalizada en milisegundos.\n\nSe atrajeron {count} fotos nuevas del sujeto.")
        self.render_page()

# ==========================================
#   APP PRINCIPAL
# ==========================================
class RoxtrosApp(ctk.CTk):
    def __init__(self):
        super().__init__(); self.title("Roxtros Enterprise - V24.0 (Armor & Logs)"); self.geometry("1200x950")
        self.aws_manager=None; self.use_cloud=False; self.db_manager=ParkDBManager()
        self.input_folder=""; self.output_folder=""; self.ref_photo_path=""
        self.clusters_found={}; self.name_entries={}; self.saved_user_names={}; self.sorted_keys_cache=[]
        self.filtered_keys_cache=[]
        self.processed_files=set(); self.face_ai=None; self.number_ai=None; self.faiss_engine=None
        self.live_monitor_active=False; self.is_paused=False; self.is_stopped=False
        self.total_scanned_count = 0; self.main_page = 0; self.main_per_page = 25 # Muestra 25 grupos por página
        
        self.ai_memory_cache = {} 
        self.cache_filename = "_roxtros_cache.pkl"
        
        self.ref_target_emb = None
        self.ref_target_hist = None
        self.ref_target_box = None
        
        self.frame_branding = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_branding.pack(pady=(15, 5), fill="x", padx=20)
        ctk.CTkLabel(self.frame_branding, text="ROXTROS", font=("Impact", 32), text_color="#2CC985").pack(side="left")
        ctk.CTkLabel(self.frame_branding, text=" // AI Sorter", font=("Arial", 16), text_color="gray").pack(side="left", padx=5, pady=(10,0))
        self.switch_cloud_var = ctk.StringVar(value="off")
        self.switch_cloud = ctk.CTkSwitch(self.frame_branding, text="Modo Nube (AWS)", command=self.toggle_source, variable=self.switch_cloud_var, onvalue="on", offvalue="off", progress_color="#E07A5F")
        self.switch_cloud.pack(side="right", padx=10)

        self.frame_top = ctk.CTkFrame(self); self.frame_top.pack(pady=10, fill="x", padx=20)
        self.pnl_local = ctk.CTkFrame(self.frame_top, fg_color="transparent"); self.pnl_local.pack(fill="x", expand=True)
        ctk.CTkButton(self.pnl_local, text="1. Carpeta FOTOS", command=self.sel_in, fg_color="#2CC985", text_color="black").pack(side="left", padx=10)
        self.lbl_in = ctk.CTkLabel(self.pnl_local, text="...", text_color="gray"); self.lbl_in.pack(side="left", padx=5)
        ctk.CTkLabel(self.pnl_local, text="|", text_color="#444").pack(side="left", padx=20)
        ctk.CTkButton(self.pnl_local, text="2. Carpeta DESTINO", command=self.sel_out, fg_color="#555555").pack(side="left", padx=10)
        self.lbl_out = ctk.CTkLabel(self.pnl_local, text="...", text_color="gray"); self.lbl_out.pack(side="left", padx=5)

        self.pnl_aws = ctk.CTkFrame(self.frame_top, fg_color="#222")
        ctk.CTkLabel(self.pnl_aws, text="CONEXIÓN SEGURA:", text_color="#E07A5F", font=("Arial", 12, "bold")).pack(side="left", padx=10)
        ctk.CTkButton(self.pnl_aws, text="🔐 Cargar Llave (.rox)", command=self.load_key, fg_color="#E07A5F", width=200).pack(side="left", padx=10)
        self.lbl_aws = ctk.CTkLabel(self.pnl_aws, text="Esperando llave...", text_color="gray"); self.lbl_aws.pack(side="left", padx=10)

        self.frame_mode = ctk.CTkFrame(self); self.frame_mode.pack(pady=5, fill="x", padx=20)
        self.mode_var = ctk.StringVar(value="Agrupar: Rostros")
        ctk.CTkSegmentedButton(self.frame_mode, values=["Agrupar: Rostros", "Agrupar: Números", "Buscar: Por Foto", "Buscar: Por RFID"], variable=self.mode_var, command=self.on_mode).pack(side="left", padx=10, fill="x", expand=True)
        
        self.frame_controls = ctk.CTkFrame(self.frame_mode, fg_color="transparent"); self.frame_controls.pack(side="right", padx=10)
        self.btn_pause = ctk.CTkButton(self.frame_controls, text="⏸️ Pausar", width=80, state="disabled", command=self.toggle_pause, fg_color="#F39C12", text_color="black"); self.btn_pause.pack(side="left", padx=5)
        self.btn_stop = ctk.CTkButton(self.frame_controls, text="⏹️ Detener", width=80, state="disabled", command=self.stop_process, fg_color="#C0392B"); self.btn_stop.pack(side="left", padx=5)
        self.btn_force_scan = ctk.CTkButton(self.frame_controls, text="🔄", width=30, fg_color="#555", command=lambda: self.run_scan(force=True))
        self.btn_force_scan.pack(side="left", padx=2)
        self.btn_scan = ctk.CTkButton(self.frame_controls, text="3. INICIAR SCAN", fg_color="#2CC985", text_color="black", command=lambda: self.run_scan(force=False)); self.btn_scan.pack(side="left", padx=2)
        
        self.params = ctk.CTkFrame(self); self.params.pack(fill="x", padx=20, pady=5)
        self.chk_live_var = ctk.BooleanVar(value=False)
        self.chk_live = ctk.CTkCheckBox(self.params, text="En Vivo", variable=self.chk_live_var, command=self.toggle_live_monitoring, text_color="cyan"); self.chk_live.pack(side="left", padx=10)
        self.entry_live_sec = ctk.CTkEntry(self.params, width=40); self.entry_live_sec.insert(0, "5"); self.entry_live_sec.pack(side="left")
        self.lbl_live_status = ctk.CTkLabel(self.params, text="[Off]", text_color="gray"); self.lbl_live_status.pack(side="left", padx=5)
        ctk.CTkLabel(self.params, text="| Sensibilidad:", text_color="#AAA").pack(side="left", padx=20)
        self.slider = ctk.CTkSlider(self.params, from_=0.3, to=0.7); self.slider.set(0.40); self.slider.pack(side="left", padx=5)
        self.lbl_slider = ctk.CTkLabel(self.params, text="0.40"); self.lbl_slider.pack(side="left")
        self.slider.configure(command=lambda v: self.lbl_slider.configure(text=f"{v:.2f}"))
        
        # --- FIX: CHECKBOX DE ROPA ---
        self.chk_clothes_var = ctk.BooleanVar(value=False)
        self.chk_clothes = ctk.CTkCheckBox(self.params, text="Ignorar Ropa (Solo Rostro)", variable=self.chk_clothes_var, text_color="orange")
        self.chk_clothes.pack(side="left", padx=15)
        # -----------------------------
        
        self.btn_ref = ctk.CTkButton(self.params, text="📸 Cargar Foto", command=self.sel_ref, fg_color="#E07A5F"); self.btn_ref.pack(side="left", padx=20)
        self.frame_rfid = ctk.CTkFrame(self.params, fg_color="transparent")
        self.entry_rfid = ctk.CTkEntry(self.frame_rfid, width=150); self.entry_rfid.pack(side="left")

        self.frame_filters = ctk.CTkFrame(self); self.frame_filters.pack(pady=5, fill="x", padx=20)
        self.lbl_stat_total = ctk.CTkLabel(self.frame_filters, text="📷 Total: 0", font=("Arial", 12, "bold")); self.lbl_stat_total.pack(side="left", padx=20)
        self.lbl_stat_groups = ctk.CTkLabel(self.frame_filters, text="👥 Grupos: 0", font=("Arial", 12, "bold"), text_color="#FFA500"); self.lbl_stat_groups.pack(side="left", padx=20)
        self.filter_var = ctk.StringVar(value="Todos")
        self.filter_seg = ctk.CTkSegmentedButton(self.frame_filters, values=["Todos", "✅ Seguro", "⚠️ Dudoso", "🔴 Revisar"], variable=self.filter_var, command=self.apply_filter, width=400)
        self.filter_seg.pack(side="right", padx=20); self.filter_seg.set("Todos")

        self.frame_nav = ctk.CTkFrame(self); self.frame_nav.pack(pady=2, fill="x", padx=20)
        self.btn_main_prev = ctk.CTkButton(self.frame_nav, text="<< Anterior", width=100, command=self.main_prev, state="disabled"); self.btn_main_prev.pack(side="left", padx=10)
        self.lbl_main_page = ctk.CTkLabel(self.frame_nav, text="Página 1", font=("Arial", 12, "bold")); self.lbl_main_page.pack(side="left", expand=True)
        self.btn_main_next = ctk.CTkButton(self.frame_nav, text="Siguiente >>", width=100, command=self.main_next, state="disabled"); self.btn_main_next.pack(side="right", padx=10)

        self.scroll_frame = ctk.CTkScrollableFrame(self, width=1000, height=400); self.scroll_frame.pack(fill="both", expand=True, padx=20, pady=5)
        self.btn_save = ctk.CTkButton(self, text="4. GUARDAR CARPETAS (COPIAR)", state="disabled", command=self.save, fg_color="green", height=40); self.btn_save.pack(pady=10, fill="x", padx=40)

    def sel_in(self): p=filedialog.askdirectory(); self.input_folder=p; self.lbl_in.configure(text=os.path.basename(p))
    def sel_out(self): p=filedialog.askdirectory(); self.output_folder=p; self.lbl_out.configure(text=os.path.basename(p))
    def sel_ref(self): 
        p=filedialog.askopenfilename(filetypes=[("Img", "*.jpg;*.png")])
        if p: self.ref_photo_path=p; self.btn_ref.configure(text="✅ Foto Lista")
    def on_mode(self, v): 
        if v=="Buscar: Por Foto": self.btn_ref.pack(side="left", padx=20)
        elif v=="Buscar: Por RFID": self.frame_rfid.pack(side="left", padx=20)
        else: self.btn_ref.pack_forget(); self.frame_rfid.pack_forget()

    def toggle_source(self):
        if self.switch_cloud_var.get() == "on":
            self.pnl_local.pack_forget(); self.pnl_aws.pack(fill="x", expand=True)
            self.use_cloud = True; self.btn_save.configure(text="4. APLICAR EN NUBE ☁️")
            logging.info("Modo cambiado a: Nube (AWS S3)")
        else:
            self.pnl_aws.pack_forget(); self.pnl_local.pack(fill="x", expand=True)
            self.use_cloud = False; self.btn_save.configure(text="4. GUARDAR CARPETAS (COPIAR)")
            logging.info("Modo cambiado a: Local (Disco Duro)")

    def load_key(self):
        p = filedialog.askopenfilename()
        if not p: return
        creds = None
        if p.endswith(".json"):
            try: creds = json.load(open(p))
            except: pass
        else:
            pwd = ctk.CTkInputDialog(text="Contraseña:", title="Seguridad").get_input()
            if pwd: creds = decrypt_data(p, pwd)
        if creds:
            msg = "Llave cargada:\n"
            if 'aws_access_key_id' in creds:
                try:
                    self.aws_manager = AWSManager(creds); self.aws_manager.s3.list_objects_v2(Bucket=self.aws_manager.bucket, MaxKeys=1)
                    msg += f"✅ AWS: {self.aws_manager.bucket}\n"
                except Exception as e: 
                    msg += f"❌ AWS Fallo: {e}\n"
            self.lbl_aws.configure(text="Llave Activa", text_color="#2CC985")
            messagebox.showinfo("Estado", msg)
        else: 
            messagebox.showerror("Error", "Fallo desencriptación")

    def load_image_unified(self, path): 
        try: 
            return Image.open(self.aws_manager.read_image_to_memory(path) if self.use_cloud and self.aws_manager else path)
        except Exception as e: 
            return None 
            
    def toggle_pause(self):
        self.is_paused=not self.is_paused
        self.btn_pause.configure(text="▶️" if self.is_paused else "⏸️")
        
    def stop_process(self): 
        self.is_stopped=True
        
    def toggle_live_monitoring(self):
        if self.chk_live_var.get(): self.live_monitor_active=True; self.lbl_live_status.configure(text="● ON", text_color="#2CC985"); threading.Thread(target=self.live_loop, daemon=True).start()
        else: self.live_monitor_active=False; self.lbl_live_status.configure(text="[Off]", text_color="gray")
    def live_loop(self):
        while self.live_monitor_active: time.sleep(int(self.entry_live_sec.get()))

    def load_cache_from_disk(self):
        if not self.input_folder: return False
        cache_path = os.path.join(self.input_folder, self.cache_filename)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.ai_memory_cache = pickle.load(f)
                return True
            except Exception as e:
                return False
        return False

    def save_cache_to_disk(self):
        if not self.input_folder or not self.ai_memory_cache: return
        cache_path = os.path.join(self.input_folder, self.cache_filename)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.ai_memory_cache, f)
        except Exception as e: pass

    def run_scan(self, force=False):
        self.is_stopped=False; self.is_paused=False
        self.btn_scan.configure(state="disabled", text="Trabajando...")
        self.btn_stop.configure(state="normal"); self.btn_pause.configure(state="normal")
        for w in self.scroll_frame.winfo_children(): w.destroy()
        self.clusters_found={}; self.name_entries={}; self.saved_user_names={}; self.sorted_keys_cache=[]
        self.filtered_keys_cache=[]
        self.processed_files=set(); self.total_scanned_count=0; self.main_page=0
        
        threading.Thread(target=self.process, args=(force,)).start()

    def process(self, force_rescan):
        try:
            if self.use_cloud: imgs = self.aws_manager.list_images()
            else: imgs = [os.path.join(r,f) for r,d,fs in os.walk(self.input_folder) for f in fs if f.lower().endswith(('jpg','png'))]
            self.total_scanned_count = len(imgs)

            loaded_from_disk = False
            if not force_rescan and not self.use_cloud:
                self.after(0, lambda: self.btn_scan.configure(text="Buscando Caché..."))
                loaded_from_disk = self.load_cache_from_disk()
            
            if force_rescan or not loaded_from_disk:
                self.ai_memory_cache.clear()
            
            needs_save = False 

            if self.mode_var.get() == "Buscar: Por Foto":
                if not self.face_ai: self.face_ai = FaceRecognitionAI()
                if not getattr(self, 'faiss_engine', None): self.faiss_engine = FaissSearchEngine()
                if not self.ref_photo_path: self.after(0,lambda:self.report_error("Falta foto de referencia")); return
                
                self.after(0, lambda: self.btn_scan.configure(text="Analizando Patrón..."))
                ref_pil = self.load_image_unified(self.ref_photo_path)
                
                if ref_pil is None:
                    self.after(0, lambda: self.report_error("❌ Error: La foto de referencia está corrupta o no se puede leer."))
                    return
                
                ref_embs, ref_boxes = self.face_ai.get_faces(ref_pil)
                if not ref_embs:
                    self.after(0, lambda: self.report_error("❌ Error: No detecté ninguna cara en la foto de referencia."))
                    return
                
                l_idx = 0; max_area = 0
                for idx, b in enumerate(ref_boxes):
                    area = (b[2]-b[0]) * (b[3]-b[1])
                    if area > max_area: max_area = area; l_idx = idx
                
                self.ref_target_emb = ref_embs[l_idx].reshape(1, -1)
                self.ref_target_box = ref_boxes[l_idx]
                self.ref_target_hist = self.face_ai.get_clothing_histogram(ref_pil, self.ref_target_box)

                matches = []
                self.after(0, lambda: self.btn_scan.configure(text="Procesando imágenes..."))
                
                for i, p in enumerate(imgs):
                    if self.is_stopped: break
                    while self.is_paused: time.sleep(0.1)

                    if p not in self.ai_memory_cache:
                        self.after(0, lambda text=f"Analizando {i}/{len(imgs)}": self.btn_scan.configure(text=text))
                        curr_pil = self.load_image_unified(p)
                        if curr_pil is None: continue 
                        
                        curr_embs, curr_boxes = self.face_ai.get_faces(curr_pil)
                        if curr_embs:
                            hists = [self.face_ai.get_clothing_histogram(curr_pil, b) for b in curr_boxes]
                            self.ai_memory_cache[p] = {'embs': curr_embs, 'boxes': curr_boxes, 'hists': hists}
                            needs_save = True
                    self.processed_files.add(p)
                    
                if self.is_stopped: return
                
                self.after(0, lambda: self.btn_scan.configure(text="Generando Index FAISS..."))
                embeddings_list = []
                paths_list = []
                for p in imgs:
                    if p in self.ai_memory_cache and self.ai_memory_cache[p].get('embs'):
                        for face_idx, emb in enumerate(self.ai_memory_cache[p]['embs']):
                            embeddings_list.append(emb)
                            paths_list.append((p, face_idx))
                            
                if embeddings_list:
                    self.faiss_engine.build_index(embeddings_list, paths_list)
                    faiss_results = self.faiss_engine.search_target(self.ref_target_emb.flatten(), k=len(embeddings_list), threshold_distance=self.slider.get() + 0.15)
                    
                    path_best_dist = {}
                    for (p, face_idx), dist in faiss_results:
                        
                        # --- FIX: IGNORAR ROPA SI ESTÁ MARCADO ---
                        if dist < 0.6 and not self.chk_clothes_var.get() and self.ref_target_hist is not None:
                            curr_hist = self.ai_memory_cache[p]['hists'][face_idx] if p in self.ai_memory_cache and 'hists' in self.ai_memory_cache[p] else None
                            if curr_hist is not None:
                                sim = self.face_ai.compare_clothing(self.ref_target_hist, curr_hist)
                                if sim < 0.2: dist += 0.3
                                elif sim > 0.6: dist -= 0.1
                        # -----------------------------------------
                        
                        if dist < self.slider.get():
                            if p not in path_best_dist or dist < path_best_dist[p]:
                                path_best_dist[p] = dist
                                
                    matches = [(p, dist) for p, dist in path_best_dist.items()]

                matches.sort(key=lambda x: x[1])
                self.clusters_found = {"Coincidencias": {'paths':[m[0] for m in matches], 'min_conf':1.0}}
                self.sorted_keys_cache = ["Coincidencias"]
            
            elif self.mode_var.get() == "Agrupar: Rostros":
                if not self.face_ai: self.face_ai = FaceRecognitionAI()
                if not getattr(self, 'faiss_engine', None): self.faiss_engine = FaissSearchEngine()
                
                all_feats = []
                metadata = []
                all_hists = [] 
                
                for i, p in enumerate(imgs):
                    if self.is_stopped: break
                    while self.is_paused: time.sleep(0.1)
                    
                    if p in self.ai_memory_cache:
                        fs = self.ai_memory_cache[p]['embs']
                        hists = self.ai_memory_cache[p].get('hists', [None]*len(fs))
                    else:
                        self.after(0, lambda text=f"Analizando {i}/{len(imgs)}": self.btn_scan.configure(text=text))
                        curr_pil = self.load_image_unified(p)
                        if curr_pil is None: continue 
                        
                        fs, boxes = self.face_ai.get_faces(curr_pil) 
                        if fs:
                            hists = [self.face_ai.get_clothing_histogram(curr_pil, b) for b in boxes]
                            self.ai_memory_cache[p] = {'embs': fs, 'boxes': boxes, 'hists': hists}
                            needs_save = True
                        else: 
                            fs = []
                            hists = []
                    
                    for f, h in zip(fs, hists): 
                        all_feats.append(f)
                        all_hists.append(h)
                        metadata.append(p)
                    self.processed_files.add(p)
                
                if all_feats:
                    self.after(0, lambda: self.btn_scan.configure(text="Generando Grafo FAISS (GPU)..."))
                    dummy_paths = [(m, i) for i, m in enumerate(metadata)] 
                    self.faiss_engine.build_index(all_feats, dummy_paths)
                    
                    k_search = min(100, len(all_feats)) 
                    data = np.array(all_feats, dtype=np.float32)
                    faiss.normalize_L2(data)
                    
                    similarities, indices = self.faiss_engine.index.search(data, k_search)
                    visited = set()
                    unique_labels = 0
                    
                    self.after(0, lambda: self.btn_scan.configure(text="Uniendo grupos (Estricto)..."))
                    for i in range(len(all_feats)):
                        if i in visited: continue
                        
                        cluster_indices = [i]
                        visited.add(i)
                        
                        for sim, neighbor_idx in zip(similarities[i], indices[i]):
                            if neighbor_idx == -1 or neighbor_idx == i: continue
                            if neighbor_idx in visited: continue
                            
                            dist = max(0.0, 1.0 - float(sim))
                            
                            # --- FIX: IGNORAR ROPA SI ESTÁ MARCADO ---
                            if dist < 0.6 and not self.chk_clothes_var.get():
                                curr_hist = all_hists[i] 
                                neigh_hist = all_hists[neighbor_idx]
                                if curr_hist is not None and neigh_hist is not None:
                                    sim_ropa = self.face_ai.compare_clothing(curr_hist, neigh_hist)
                                    if sim_ropa < 0.2: dist += 0.3
                                    elif sim_ropa > 0.6: dist -= 0.1
                            # -----------------------------------------
                            
                            if dist < self.slider.get():
                                visited.add(neighbor_idx)
                                cluster_indices.append(neighbor_idx)
                        
                        k_name = f"Participante_{unique_labels+1:02d}"
                        cluster_paths = list(dict.fromkeys([metadata[idx] for idx in cluster_indices])) 
                        
                        score = 1.0
                        if len(cluster_indices) > 1:
                            vecs = [all_feats[idx] for idx in cluster_indices]
                            centroid = np.mean(vecs, axis=0).reshape(1, -1)
                            avg_dist = np.mean(cosine_distances(vecs, centroid))
                            score = max(0.0, 1.0 - (avg_dist * 2.5))
                        
                        self.clusters_found.setdefault(k_name, {'paths': cluster_paths, 'min_conf': score})
                        unique_labels += 1
                        
                    self.sorted_keys_cache = sorted(self.clusters_found.keys())

            elif self.mode_var.get() == "Agrupar: Números":
                if not self.number_ai: self.number_ai = NumberRecognitionAI()
                for i, p in enumerate(imgs):
                    if self.is_stopped: break
                    while self.is_paused: time.sleep(0.1)
                    self.after(0, lambda text=f"OCR {i}/{len(imgs)}": self.btn_scan.configure(text=text))
                    
                    curr_pil = self.load_image_unified(p)
                    if curr_pil is None: continue 
                    
                    dets = self.number_ai.get_numbers(curr_pil)
                    if dets:
                        for (txt, conf) in dets:
                            self.clusters_found.setdefault(txt, {'paths':[], 'min_conf':conf})['paths'].append(p)
                    else:
                        self.clusters_found.setdefault("Sin_Numero", {'paths':[], 'min_conf':0.0})['paths'].append(p)
                self.sorted_keys_cache = sorted(self.clusters_found.keys())

            if needs_save and not self.use_cloud:
                self.after(0, lambda: self.btn_scan.configure(text="Guardando Caché..."))
                self.save_cache_to_disk()

            self.after(0, lambda: self.apply_filter("Todos"))
            self.after(0, lambda: self.btn_scan.configure(state="normal", text="3. INICIAR SCAN"))
            self.after(0, lambda: self.btn_save.configure(state="normal"))
            
        except Exception as e:
            self.after(0, lambda err=e: self.report_error(f"Error Crítico: Revisa roxtros_system.log para detalles."))

    # --- RENDERIZADO ---
    def apply_filter(self, dummy=None):
        mode = self.filter_var.get()
        if mode == "Todos": self.filtered_keys_cache = self.sorted_keys_cache
        else:
            self.filtered_keys_cache = []
            for k in self.sorted_keys_cache:
                conf = self.clusters_found[k].get('min_conf', 1.0)
                if mode == "✅ Seguro" and conf >= 0.8: self.filtered_keys_cache.append(k)
                elif mode == "⚠️ Dudoso" and 0.5 <= conf < 0.8: self.filtered_keys_cache.append(k)
                elif mode == "🔴 Revisar" and conf < 0.5: self.filtered_keys_cache.append(k)
        self.main_page = 0; self.render_grid()

    def render_grid(self):
        for w in self.scroll_frame.winfo_children(): w.destroy()
        start = self.main_page * self.main_per_page
        keys = self.filtered_keys_cache[start : start+self.main_per_page]
        
        self.lbl_stat_total.configure(text=f"📷 Total: {self.total_scanned_count}")
        self.lbl_stat_groups.configure(text=f"👥 Grupos: {len(self.filtered_keys_cache)}")
        
        # --- FIX: LÓGICA DE PAGINACIÓN ---
        total_pages = max(1, ((len(self.filtered_keys_cache) - 1) // self.main_per_page) + 1)
        self.lbl_main_page.configure(text=f"Página {self.main_page + 1} de {total_pages}")
        
        if self.main_page > 0: self.btn_main_prev.configure(state="normal")
        else: self.btn_main_prev.configure(state="disabled")
        
        if self.main_page < total_pages - 1: self.btn_main_next.configure(state="normal")
        else: self.btn_main_next.configure(state="disabled")
        # ---------------------------------
        
        threading.Thread(target=self.render_rows, args=(keys,)).start()
        self.frame_rows = {} 
        for i, k in enumerate(keys):
            paths = self.clusters_found[k]['paths']
            conf = self.clusters_found[k].get('min_conf', 1.0)
            if conf >= 0.8: border_col = "#2CC985" 
            elif conf >= 0.5: border_col = "#F1C40F"
            else: border_col = "#E74C3C" 
            f = ctk.CTkFrame(self.scroll_frame, border_width=2, border_color=border_col)
            f.pack(fill="x", pady=2, padx=5)
            self.frame_rows[k] = f
            self.lbl_thumb = ctk.CTkLabel(f, text="[...]", width=110); self.lbl_thumb.pack(side="left", padx=5)
            ctk.CTkLabel(f, text=f"{len(paths)} fotos", width=60).pack(side="left")
            ent = ctk.CTkEntry(f, width=200); ent.insert(0, self.saved_user_names.get(k, k))
            ent.pack(side="left", padx=10, fill="x", expand=True)
            ent.bind("<FocusOut>", lambda e, k=k, entry=ent: self.upd_name(k, entry.get()))
            ctk.CTkButton(f, text="📷 Ver", width=80, command=lambda p=paths, k=k: GalleryPopup(self, p, k)).pack(side="right", padx=10)

    def render_rows(self, keys):
        for k in keys:
            try:
                paths = self.clusters_found[k]['paths']
                if paths:
                    pil = self.load_image_unified(paths[0])
                    if pil:
                        pil.thumbnail((110,110))
                        ci = ctk.CTkImage(light_image=pil, dark_image=pil, size=(110,110))
                        self.after(0, lambda fr=self.frame_rows.get(k), img=ci: self.update_row_icon(fr, img))
            except: pass

    def update_row_icon(self, frame, img):
        if frame and frame.winfo_exists():
            for child in frame.winfo_children():
                if isinstance(child, ctk.CTkLabel) and child.cget("text") == "[...]": child.configure(image=img, text="")

    def upd_name(self, k, v): self.saved_user_names[k] = v
    def get_existing_group_names(self): return list(self.saved_user_names.values())
    
    def main_prev(self): 
        if self.main_page>0: self.main_page-=1; self.render_grid()
    def main_next(self): self.main_page+=1; self.render_grid()

    def save(self):
        if self.use_cloud and not messagebox.askyesno("Confirmar", "Se alterará S3. ¿Seguro?"): return
        threading.Thread(target=self.save_thread).start()
    
    def save_thread(self):
        c = 0
        logging.info("Iniciando guardado de carpetas/S3...")
        for k, data in self.clusters_found.items():
            name = self.saved_user_names.get(k, k)
            safe = "".join([char for char in name if char.isalnum() or char in (' ','_')]).strip()
            
            # --- FIX: COPIAR EN VEZ DE MOVER ---
            for p in data['paths']:
                try:
                    if self.use_cloud:
                        nk = f"{safe}/{p.split('/')[-1]}"
                        if self.aws_manager.copy_object(p, nk): 
                            self.aws_manager.delete_object(p); c+=1
                    else:
                        d = os.path.join(self.output_folder, safe); os.makedirs(d, exist_ok=True)
                        shutil.copy2(p, os.path.join(d, os.path.basename(p))) 
                        c += 1
                except Exception as e: 
                    logging.error(f"Error copiando foto {p} a {safe}: {e}")
            # -----------------------------------
        
        logging.info(f"Guardado finalizado. {c} fotos procesadas.")
        self.after(0, lambda: messagebox.showinfo("Fin", f"Procesados: {c}")); self.after(0, lambda: self.btn_save.configure(state="normal"))
    
    def report_error(self, m): 
        messagebox.showerror("Error", m)
        self.btn_scan.configure(state="normal")