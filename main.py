import cv2
import numpy as np
import mss
import time
import os
import sys
import threading
import tkinter as tk
import json

# --- СИСТЕМНЫЕ ПУТИ ---
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

data_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "data.txt")
settings_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "settings.json")
presets_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "presets.json")

# --- НАСТРОЙКИ ЗОН ---
MONITOR_COORDS = {"top": 100, "left": 5, "width": 750, "height": 40}
MONITOR_FACING = {"top": 185, "left": 520, "width": 130, "height": 30}

is_running = False
templates = []

# Размер интерфейса
ui_scale = "big"  # mini, def, big, auto

# --- ЗАГРУЗКА ШАБЛОНОВ ---
def load_templates(scale="big"):
    """Загрузка шаблонов из выбранной папки"""
    global templates
    templates = []
    
    tpl_dir = resource_path(f'templates_{scale}')
    if os.path.exists(tpl_dir):
        for f in os.listdir(tpl_dir):
            if f.endswith('.png'):
                name = f.replace('.png', '')
                sym = {"dot": ".", "minus": "-", "slash": "/"}.get(name, name)
                try:
                    img = cv2.imdecode(np.fromfile(os.path.join(tpl_dir, f), np.uint8),
                                       cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                        templates.append((sym, img))
                except:
                    pass
        print(f"[INFO] Loaded {len(templates)} templates from templates_{scale}")
    else:
        print(f"[WARNING] Directory templates_{scale} not found!")

# Загружаем шаблоны по умолчанию
load_templates(ui_scale)

# --- OCR ---
def get_text(sct, area):
    scr = np.array(sct.grab(area))
    gray = cv2.cvtColor(scr, cv2.COLOR_BGRA2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    matches = []
    for s, temp in templates:
        res = cv2.matchTemplate(binary, temp, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= (0.98 if s in [".", "-"] else 0.92))
        for pt in zip(*loc[::-1]):
            matches.append({'x': pt[0], 's': s})
    matches.sort(key=lambda x: x['x'])
    res_str, last_x = "", -100
    for m in matches:
        if m['x'] >= last_x + 3:
            res_str += m['s']
            last_x = m['x']
    return res_str

# --- СКАН ---
def scan_logic():
    global is_running
    with mss.mss() as sct:
        while is_running:
            c_str = get_text(sct, MONITOR_COORDS)
            if c_str.count('/') == 2:
                try:
                    p = c_str.lstrip('.').split('/')
                    with open(data_path, "w") as f:
                        f.write(f"{p[0]},{p[1]},{p[2]}")
                    label_coords.config(
                        text=f"{float(p[0]):.3f} | {float(p[1]):.5f} | {float(p[2]):.3f}",
                        fg="#00ff00")
                except:
                    pass

            if face_var.get():
                f_str = get_text(sct, MONITOR_FACING)
                if f_str:
                    try:
                        yaw = f_str.replace('(', '').split('/')[0]
                        face_label.config(text=f"{float(yaw):.1f}",
                                          fg=ent_face_color.get())
                        
                        # Применяем сохраненную геометрию при первом показе
                        global face_window_geometry_applied
                        if not face_window_geometry_applied and "face_window_geometry" in settings_data:
                            face_window.geometry(settings_data["face_window_geometry"])
                            face_window_geometry_applied = True
                        
                        face_window.deiconify()
                    except:
                        pass
            else:
                face_window.withdraw()
                face_window_geometry_applied = False  # Сбрасываем флаг когда окно скрывается

            time.sleep(0.02)

# --- МАТЕМАТИКА ---
def calculate_axis_offset(coord, c1, c2):
    """Вычисляет знаковое расстояние по одной оси"""
    min_c, max_c = min(c1, c2), max(c1, c2)
    if min_c <= coord <= max_c:
        # Внутри - положительное расстояние до ближайшего края
        return min(coord - min_c, max_c - coord)
    else:
        # Снаружи - отрицательное расстояние
        return -max(min_c - coord, coord - max_c)

def calculate_signed_offset(cx, cz, x1, z1, x2, z2):
    """Вычисляет общее знаковое расстояние до прямоугольника"""
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_z, max_z = min(z1, z2), max(z1, z2)
    dx = max(0, min_x - cx, cx - max_x)
    dz = max(0, min_z - cz, cz - max_z)
    d = (dx ** 2 + dz ** 2) ** 0.5
    return -d if d > 0 else min(cx - min_x, max_x - cx, cz - min_z, max_z - cz)

def calculate_offset_with_components(cx, cz, x1, z1, x2, z2):
    """Вычисляет общий offset и компоненты X, Z для Z-Neo"""
    x_off = calculate_axis_offset(cx, x1, x2)
    z_off = calculate_axis_offset(cz, z1, z2)
    
    # Логика расчёта общего offset:
    if x_off >= 0 and z_off >= 0:
        # Оба положительные или на границе - положительный sqrt
        total = (x_off ** 2 + z_off ** 2) ** 0.5
    elif x_off < 0 and z_off < 0:
        # Оба отрицательные - отрицательный sqrt
        total = -(((-x_off) ** 2 + (-z_off) ** 2) ** 0.5)
    else:
        # Разные знаки - берём отрицательное значение (то что с минусом)
        total = min(x_off, z_off)
    
    return total, x_off, z_off

stored_z = None

# История координат для Delta (последние 3 тика)
coord_history = []
prev_cy = None

# Глобальные настройки для доступа из toggle функций
settings_data = {}

# Флаги для отслеживания применения геометрии окон
face_window_geometry_applied = False

# Флаги для Delta триггеров
delta_y1_triggered = False
delta_y2_triggered = False
delta_mm_triggered = False

def update_calc():
    global stored_z, coord_history, prev_cy, delta_y1_triggered, delta_y2_triggered, delta_mm_triggered
    
    if os.path.exists(data_path):
        try:
            with open(data_path) as f:
                cx, cy, cz = map(float, f.read().split(','))
            
            # ВАЖНО: Обновляем историю только когда координаты ИЗМЕНИЛИСЬ (новый тик Minecraft)
            # update_calc вызывается каждые 10мс, но тик Minecraft = 50мс
            # Без этой проверки история заполнится дубликатами одного тика
            if len(coord_history) == 0 or cy != coord_history[-1][1]:
                coord_history.append((cx, cy, cz))
                if len(coord_history) > 5:  # Храним 5 тиков
                    coord_history.pop(0)
            
            y1, lim = float(ent_y.get()), float(ent_limit.get())
            
            # ВАЖНО: Delta срабатывает при пересечении границы ВНИЗ
            # Сброс триггеров когда пересекли границу ВВЕРХ (новый прыжок)
            if prev_cy is not None:
                if cy > y1 and prev_cy <= y1:
                    delta_y1_triggered = False
                
                if neo_var.get():
                    y2 = float(ent_y2.get())
                    if cy > y2 and prev_cy <= y2:
                        delta_y2_triggered = False
                
                if show_mm_offset_var.get() or show_mm_x_offset_var.get() or show_mm_z_offset_var.get():
                    mm_y = float(ent_mm_y.get())
                    if cy > mm_y and prev_cy <= mm_y:
                        delta_mm_triggered = False
            
            # Проверяем игнорирование осей
            ignore_x = ignore_x_var.get()
            ignore_z = ignore_z_var.get()

            # --- ОБЫЧНЫЙ OFFSET ---
            if neo_var.get():
                y2 = float(ent_y2.get())
                
                # Обработка Y2 (для stored_z)
                if use_delta_y2_var.get():
                    # Delta режим Y2: пересечение Y2 вниз, берём Z СТРОГО 2 тика назад
                    if prev_cy is not None and prev_cy > y2 and cy <= y2 and not delta_y2_triggered:
                        delta_y2_triggered = True
                        history_len = len(coord_history)
                        
                        print(f"\n=== Y2 TRIGGER at y={cy:.3f}, prev_y={prev_cy:.3f} ===")
                        print(f"History length: {history_len}")
                        for i, (hx, hy, hz) in enumerate(coord_history):
                            print(f"  [{i}] y={hy:.3f}, z={hz:.3f}")
                        
                        # coord_history[-1] = ТЕКУЩИЙ тик, [-2] = 1т назад, [-3] = 2т назад
                        # ВАЖНО: Берём ТОЛЬКО [-3] (2 тика назад), без fallback!
                        if history_len >= 3:
                            stored_z = coord_history[-3][2]  # Z 2 тика назад
                            print(f"✓ Using [-3]: stored_z={stored_z:.3f} (y={coord_history[-3][1]:.3f})")
                        else:
                            # Если истории недостаточно - НЕ сохраняем stored_z (пропускаем этот прыжок)
                            stored_z = None
                            print(f"✗ INSUFFICIENT HISTORY: need 3, have {history_len}. Skipping this jump.")
                        print(f"==================\n")
                else:
                    # Обычный режим Y2 - точное совпадение высоты
                    if abs(cy - y2) < 0.005:
                        stored_z = cz
                
                # Обработка Y1 (для расчёта offset)
                if use_delta_y1_var.get():
                    # Delta режим Y1: пересечение Y1 вниз, берём X 1 тик назад
                    if prev_cy is not None and prev_cy > y1 and cy <= y1 and not delta_y1_triggered and stored_z is not None:
                        delta_y1_triggered = True
                        
                        print(f"\n=== Y1 TRIGGER at y={cy:.3f}, prev_y={prev_cy:.3f} ===")
                        print(f"stored_z (from Y2): {stored_z:.3f}")
                        print(f"History length: {len(coord_history)}")
                        
                        # Берём X из 1 тика назад: coord_history[-2]
                        if len(coord_history) >= 2:
                            cx_calc = coord_history[-2][0]  # X 1 тик назад
                            print(f"✓ Using [-2] for X: cx_calc={cx_calc:.3f} (y={coord_history[-2][1]:.3f}, z={coord_history[-2][2]:.3f})")
                        else:
                            cx_calc = cx
                            print(f"! NO HISTORY: cx_calc={cx_calc:.3f} (current)")
                        print(f"Will calculate offset with X={cx_calc:.3f}, Z={stored_z:.3f}")
                        print(f"==================\n")
                        
                        if not ignore_x and not ignore_z:
                            total_off, x_off, z_off = calculate_offset_with_components(
                                cx_calc, stored_z,
                                float(ent_x1.get()), float(ent_z1.get()),
                                float(ent_x2.get()), float(ent_z2.get()))
                        elif ignore_x and not ignore_z:
                            z_off = calculate_axis_offset(stored_z, float(ent_z1.get()), float(ent_z2.get()))
                            total_off = abs(z_off) if z_off >= 0 else -abs(z_off)
                            x_off = 0
                        elif ignore_z and not ignore_x:
                            x_off = calculate_axis_offset(cx_calc, float(ent_x1.get()), float(ent_x2.get()))
                            total_off = abs(x_off) if x_off >= 0 else -abs(x_off)
                            z_off = 0
                        else:
                            total_off = x_off = z_off = 0
                        
                        # В Delta режиме показываем offset'ы в пределах лимита
                        if abs(total_off) <= lim:
                            if not ignore_x and not ignore_z:
                                show_res(total_off)
                                show_x_offset(x_off)
                                show_z_offset(z_off)
                            elif not ignore_z:
                                show_res(total_off)
                                show_z_offset(z_off)
                            elif not ignore_x:
                                show_res(total_off)
                                show_x_offset(x_off)
                        stored_z = None
                else:
                    # Обычный режим
                    if abs(cy - y1) < 0.005 and stored_z is not None:
                        if not ignore_x and not ignore_z:
                            total_off, x_off, z_off = calculate_offset_with_components(
                                cx, stored_z,
                                float(ent_x1.get()), float(ent_z1.get()),
                                float(ent_x2.get()), float(ent_z2.get()))
                        elif ignore_x and not ignore_z:
                            z_off = calculate_axis_offset(stored_z, float(ent_z1.get()), float(ent_z2.get()))
                            total_off = abs(z_off) if z_off >= 0 else -abs(z_off)
                            x_off = 0
                        elif ignore_z and not ignore_x:
                            x_off = calculate_axis_offset(cx, float(ent_x1.get()), float(ent_x2.get()))
                            total_off = abs(x_off) if x_off >= 0 else -abs(x_off)
                            z_off = 0
                        else:
                            total_off = x_off = z_off = 0
                        
                        if abs(total_off) <= lim:
                            if not ignore_x and not ignore_z:
                                show_res(total_off)
                                show_x_offset(x_off)
                                show_z_offset(z_off)
                            elif not ignore_z:
                                show_res(total_off)
                                show_z_offset(z_off)
                            elif not ignore_x:
                                show_res(total_off)
                                show_x_offset(x_off)
                        stored_z = None
            else:
                # Без Z-Neo
                if use_delta_y1_var.get():
                    # Delta режим (<=)
                    if prev_cy is not None and prev_cy > y1 and cy <= y1 and not delta_y1_triggered:
                        delta_y1_triggered = True
                        # Берём координаты 1 тик назад
                        if len(coord_history) >= 2:
                            cx_calc, _, cz_calc = coord_history[-2]
                        else:
                            cx_calc, cz_calc = cx, cz
                        
                        if not ignore_x and not ignore_z:
                            total_off, x_off, z_off = calculate_offset_with_components(
                                cx_calc, cz_calc,
                                float(ent_x1.get()), float(ent_z1.get()),
                                float(ent_x2.get()), float(ent_z2.get()))
                        elif ignore_x and not ignore_z:
                            z_off = calculate_axis_offset(cz_calc, float(ent_z1.get()), float(ent_z2.get()))
                            total_off = abs(z_off) if z_off >= 0 else -abs(z_off)
                            x_off = 0
                        elif ignore_z and not ignore_x:
                            x_off = calculate_axis_offset(cx_calc, float(ent_x1.get()), float(ent_x2.get()))
                            total_off = abs(x_off) if x_off >= 0 else -abs(x_off)
                            z_off = 0
                        else:
                            total_off = x_off = z_off = 0
                        
                        # В Delta режиме проверяем лимит
                        if abs(total_off) <= lim:
                            if not ignore_x and not ignore_z:
                                show_res(total_off)
                                show_x_offset(x_off)
                                show_z_offset(z_off)
                            elif not ignore_z:
                                show_res(total_off)
                                show_z_offset(z_off)
                            elif not ignore_x:
                                show_res(total_off)
                                show_x_offset(x_off)
                else:
                    # Обычный режим
                    if abs(cy - y1) < 0.005:
                        if not ignore_x and not ignore_z:
                            total_off, x_off, z_off = calculate_offset_with_components(
                                cx, cz,
                                float(ent_x1.get()), float(ent_z1.get()),
                                float(ent_x2.get()), float(ent_z2.get()))
                        elif ignore_x and not ignore_z:
                            z_off = calculate_axis_offset(cz, float(ent_z1.get()), float(ent_z2.get()))
                            total_off = abs(z_off) if z_off >= 0 else -abs(z_off)
                            x_off = 0
                        elif ignore_z and not ignore_x:
                            x_off = calculate_axis_offset(cx, float(ent_x1.get()), float(ent_x2.get()))
                            total_off = abs(x_off) if x_off >= 0 else -abs(x_off)
                            z_off = 0
                        else:
                            total_off = x_off = z_off = 0
                        
                        if abs(total_off) <= lim:
                            if not ignore_x and not ignore_z:
                                show_res(total_off)
                                show_x_offset(x_off)
                                show_z_offset(z_off)
                            elif not ignore_z:
                                show_res(total_off)
                                show_z_offset(z_off)
                            elif not ignore_x:
                                show_res(total_off)
                                show_x_offset(x_off)

            # --- MM OFFSET ---
            if show_mm_offset_var.get() or show_mm_x_offset_var.get() or show_mm_z_offset_var.get():
                mm_y = float(ent_mm_y.get())
                mm_lim = float(ent_mm_limit.get())
                
                ignore_x_mm = ignore_x_mm_var.get()
                ignore_z_mm = ignore_z_mm_var.get()
                
                if use_delta_mm_var.get():
                    # Delta режим (<=)
                    if prev_cy is not None and prev_cy > mm_y and cy <= mm_y and not delta_mm_triggered:
                        delta_mm_triggered = True
                        # Берём координаты 1 тик назад
                        if len(coord_history) >= 2:
                            cx_calc, _, cz_calc = coord_history[-2]
                        else:
                            cx_calc, cz_calc = cx, cz
                        
                        if not ignore_x_mm and not ignore_z_mm:
                            mm_total, mm_x, mm_z = calculate_offset_with_components(
                                cx_calc, cz_calc,
                                float(ent_x1_mm.get()), float(ent_z1_mm.get()),
                                float(ent_x2_mm.get()), float(ent_z2_mm.get()))
                        elif ignore_x_mm and not ignore_z_mm:
                            mm_z = calculate_axis_offset(cz_calc, float(ent_z1_mm.get()), float(ent_z2_mm.get()))
                            mm_total = abs(mm_z) if mm_z >= 0 else -abs(mm_z)
                            mm_x = 0
                        elif ignore_z_mm and not ignore_x_mm:
                            mm_x = calculate_axis_offset(cx_calc, float(ent_x1_mm.get()), float(ent_x2_mm.get()))
                            mm_total = abs(mm_x) if mm_x >= 0 else -abs(mm_x)
                            mm_z = 0
                        else:
                            mm_total = mm_x = mm_z = 0
                        
                        # В Delta режиме проверяем лимит
                        if abs(mm_total) <= mm_lim:
                            if not ignore_x_mm and not ignore_z_mm:
                                show_mm_res(mm_total)
                                show_mm_x_offset(mm_x)
                                show_mm_z_offset(mm_z)
                            elif not ignore_z_mm:
                                show_mm_res(mm_total)
                                show_mm_z_offset(mm_z)
                            elif not ignore_x_mm:
                                show_mm_res(mm_total)
                                show_mm_x_offset(mm_x)
                else:
                    # Обычный режим
                    if abs(cy - mm_y) < 0.005:
                        if not ignore_x_mm and not ignore_z_mm:
                            mm_total, mm_x, mm_z = calculate_offset_with_components(
                                cx, cz,
                                float(ent_x1_mm.get()), float(ent_z1_mm.get()),
                                float(ent_x2_mm.get()), float(ent_z2_mm.get()))
                        elif ignore_x_mm and not ignore_z_mm:
                            mm_z = calculate_axis_offset(cz, float(ent_z1_mm.get()), float(ent_z2_mm.get()))
                            mm_total = abs(mm_z) if mm_z >= 0 else -abs(mm_z)
                            mm_x = 0
                        elif ignore_z_mm and not ignore_x_mm:
                            mm_x = calculate_axis_offset(cx, float(ent_x1_mm.get()), float(ent_x2_mm.get()))
                            mm_total = abs(mm_x) if mm_x >= 0 else -abs(mm_x)
                            mm_z = 0
                        else:
                            mm_total = mm_x = mm_z = 0
                        
                        if abs(mm_total) <= mm_lim:
                            if not ignore_x_mm and not ignore_z_mm:
                                show_mm_res(mm_total)
                                show_mm_x_offset(mm_x)
                                show_mm_z_offset(mm_z)
                            elif not ignore_z_mm:
                                show_mm_res(mm_total)
                                show_mm_z_offset(mm_z)
                            elif not ignore_x_mm:
                                show_mm_res(mm_total)
                                show_mm_x_offset(mm_x)
            
            # ВАЖНО: Обновляем prev_cy в САМОМ КОНЦЕ, после всех проверок
            prev_cy = cy
                        
        except:
            pass
    root.after(10, update_calc)

# --- СОХРАНЕНИЕ И ЗАГРУЗКА НАСТРОЕК ---
def save_settings():
    """Сохранение всех настроек в JSON"""
    settings = {
        # Основные координаты
        "x1": ent_x1.get(),
        "z1": ent_z1.get(),
        "x2": ent_x2.get(),
        "z2": ent_z2.get(),
        "y1": ent_y.get(),
        "y2": ent_y2.get(),
        "limit": ent_limit.get(),
        
        # MM координаты
        "x1_mm": ent_x1_mm.get(),
        "z1_mm": ent_z1_mm.get(),
        "x2_mm": ent_x2_mm.get(),
        "z2_mm": ent_z2_mm.get(),
        "mm_y": ent_mm_y.get(),
        "mm_limit": ent_mm_limit.get(),
        
        # Facing color
        "face_color": ent_face_color.get(),
        
        # UI Scale
        "ui_scale": ui_scale,
        
        # Чекбоксы
        "neo_enabled": neo_var.get(),
        "face_enabled": face_var.get(),
        "show_coords_area": show_coords_area.get(),
        "show_facing_area": show_facing_area.get(),
        "show_offset": show_offset_var.get(),
        "show_x_offset": show_x_offset_var.get(),
        "show_z_offset": show_z_offset_var.get(),
        "show_mm_offset": show_mm_offset_var.get(),
        "show_mm_x_offset": show_mm_x_offset_var.get(),
        "show_mm_z_offset": show_mm_z_offset_var.get(),
        
        # Delta чекбоксы
        "use_delta_y1": use_delta_y1_var.get(),
        "use_delta_y2": use_delta_y2_var.get(),
        "use_delta_mm": use_delta_mm_var.get(),
        
        # Ignore чекбоксы
        "ignore_x": ignore_x_var.get(),
        "ignore_z": ignore_z_var.get(),
        "ignore_x_mm": ignore_x_mm_var.get(),
        "ignore_z_mm": ignore_z_mm_var.get(),
        
        # Зоны сканирования
        "monitor_coords": MONITOR_COORDS.copy(),
        "monitor_facing": MONITOR_FACING.copy()
    }
    
    # Сохраняем позиции и видимость offset окон
    if res_window.winfo_viewable():
        settings["offset_window_geometry"] = res_window.geometry()
    if x_offset_window.winfo_viewable():
        settings["x_offset_window_geometry"] = x_offset_window.geometry()
    if z_offset_window.winfo_viewable():
        settings["z_offset_window_geometry"] = z_offset_window.geometry()
    if mm_offset_window.winfo_viewable():
        settings["mm_offset_window_geometry"] = mm_offset_window.geometry()
    if mm_x_offset_window.winfo_viewable():
        settings["mm_x_offset_window_geometry"] = mm_x_offset_window.geometry()
    if mm_z_offset_window.winfo_viewable():
        settings["mm_z_offset_window_geometry"] = mm_z_offset_window.geometry()
    if face_window.winfo_viewable():
        settings["face_window_geometry"] = face_window.geometry()
    
    settings_data = settings
    
    try:
        with open(settings_path, 'w') as f:
            json.dump(settings_data, f, indent=2)
    except:
        pass

def load_settings():
    """Загрузка настроек из JSON"""
    global MONITOR_COORDS, MONITOR_FACING, settings_data
    
    if not os.path.exists(settings_path):
        return
    
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        # Сохраняем настройки глобально
        settings_data = settings
        
        # Основные координаты
        ent_x1.delete(0, tk.END)
        ent_x1.insert(0, settings.get("x1", "0"))
        ent_z1.delete(0, tk.END)
        ent_z1.insert(0, settings.get("z1", "0"))
        ent_x2.delete(0, tk.END)
        ent_x2.insert(0, settings.get("x2", "1"))
        ent_z2.delete(0, tk.END)
        ent_z2.insert(0, settings.get("z2", "1"))
        ent_y.delete(0, tk.END)
        ent_y.insert(0, settings.get("y1", "122.10214"))
        ent_y2.delete(0, tk.END)
        ent_y2.insert(0, settings.get("y2", "123.000"))
        ent_limit.delete(0, tk.END)
        ent_limit.insert(0, settings.get("limit", "1.000"))
        
        # MM координаты
        ent_x1_mm.delete(0, tk.END)
        ent_x1_mm.insert(0, settings.get("x1_mm", "0"))
        ent_z1_mm.delete(0, tk.END)
        ent_z1_mm.insert(0, settings.get("z1_mm", "0"))
        ent_x2_mm.delete(0, tk.END)
        ent_x2_mm.insert(0, settings.get("x2_mm", "1"))
        ent_z2_mm.delete(0, tk.END)
        ent_z2_mm.insert(0, settings.get("z2_mm", "1"))
        ent_mm_y.delete(0, tk.END)
        ent_mm_y.insert(0, settings.get("mm_y", "122.10214"))
        ent_mm_limit.delete(0, tk.END)
        ent_mm_limit.insert(0, settings.get("mm_limit", "1.000"))
        
        # Facing color
        ent_face_color.delete(0, tk.END)
        ent_face_color.insert(0, settings.get("face_color", "#FFFF00"))
        
        # UI Scale
        global ui_scale
        loaded_scale = settings.get("ui_scale", "big")
        if loaded_scale != ui_scale:
            ui_scale = loaded_scale
            ui_scale_var.set(loaded_scale)
            load_templates(loaded_scale)
        
        # Чекбоксы
        neo_var.set(settings.get("neo_enabled", False))
        face_var.set(settings.get("face_enabled", False))
        show_coords_area.set(settings.get("show_coords_area", False))
        show_facing_area.set(settings.get("show_facing_area", False))
        show_offset_var.set(settings.get("show_offset", False))
        show_x_offset_var.set(settings.get("show_x_offset", False))
        show_z_offset_var.set(settings.get("show_z_offset", False))
        show_mm_offset_var.set(settings.get("show_mm_offset", False))
        show_mm_x_offset_var.set(settings.get("show_mm_x_offset", False))
        show_mm_z_offset_var.set(settings.get("show_mm_z_offset", False))
        
        # Delta чекбоксы
        use_delta_y1_var.set(settings.get("use_delta_y1", False))
        use_delta_y2_var.set(settings.get("use_delta_y2", False))
        use_delta_mm_var.set(settings.get("use_delta_mm", False))
        
        # Ignore чекбоксы
        ignore_x_var.set(settings.get("ignore_x", False))
        ignore_z_var.set(settings.get("ignore_z", False))
        ignore_x_mm_var.set(settings.get("ignore_x_mm", False))
        ignore_z_mm_var.set(settings.get("ignore_z_mm", False))
        
        # Зоны сканирования
        if "monitor_coords" in settings:
            MONITOR_COORDS.update(settings["monitor_coords"])
            update_overlay(coords_overlay, MONITOR_COORDS)
        if "monitor_facing" in settings:
            MONITOR_FACING.update(settings["monitor_facing"])
            update_overlay(facing_overlay, MONITOR_FACING)
        
        # Применяем состояния UI
        if neo_var.get():
            toggle_neo_ui()
        if show_coords_area.get():
            toggle_coords_area()
        if show_facing_area.get():
            toggle_facing_area()
        if show_mm_offset_var.get() or show_mm_x_offset_var.get() or show_mm_z_offset_var.get():
            toggle_mm_offset_ui()
        
        # Восстанавливаем позиции и видимость offset окон
        if "offset_window_geometry" in settings and show_offset_var.get():
            res_window.geometry(settings["offset_window_geometry"])
            res_window.deiconify()
        if "x_offset_window_geometry" in settings and show_x_offset_var.get():
            x_offset_window.geometry(settings["x_offset_window_geometry"])
            x_offset_window.deiconify()
        if "z_offset_window_geometry" in settings and show_z_offset_var.get():
            z_offset_window.geometry(settings["z_offset_window_geometry"])
            z_offset_window.deiconify()
        if "mm_offset_window_geometry" in settings and show_mm_offset_var.get():
            mm_offset_window.geometry(settings["mm_offset_window_geometry"])
            mm_offset_window.deiconify()
        if "mm_x_offset_window_geometry" in settings and show_mm_x_offset_var.get():
            mm_x_offset_window.geometry(settings["mm_x_offset_window_geometry"])
            mm_x_offset_window.deiconify()
        if "mm_z_offset_window_geometry" in settings and show_mm_z_offset_var.get():
            mm_z_offset_window.geometry(settings["mm_z_offset_window_geometry"])
            mm_z_offset_window.deiconify()
        if "face_window_geometry" in settings and face_var.get():
            face_window.geometry(settings["face_window_geometry"])
            face_window.deiconify()
            
    except:
        pass

# --- PRESETS (БЫСТРЫЕ НАСТРОЙКИ) ---
import tkinter.simpledialog
import tkinter.messagebox

def save_preset():
    """Сохранение текущих координат как preset"""
    preset_name = tk.simpledialog.askstring("Save Preset", "Enter preset name:")
    if not preset_name:
        return
    
    preset_data = {
        "x1": ent_x1.get(),
        "z1": ent_z1.get(),
        "x2": ent_x2.get(),
        "z2": ent_z2.get(),
        "y1": ent_y.get(),
        "y2": ent_y2.get(),
        "limit": ent_limit.get(),
        "x1_mm": ent_x1_mm.get(),
        "z1_mm": ent_z1_mm.get(),
        "x2_mm": ent_x2_mm.get(),
        "z2_mm": ent_z2_mm.get(),
        "mm_y": ent_mm_y.get(),
        "mm_limit": ent_mm_limit.get(),
    }
    
    try:
        if os.path.exists(presets_path):
            with open(presets_path, 'r') as f:
                presets = json.load(f)
        else:
            presets = {}
        
        presets[preset_name] = preset_data
        
        with open(presets_path, 'w') as f:
            json.dump(presets, f, indent=2)
        
        tk.messagebox.showinfo("Success", f"Preset '{preset_name}' saved!")
    except Exception as e:
        tk.messagebox.showerror("Error", f"Error saving preset: {e}")

def load_preset():
    """Показать список presets и загрузить выбранный"""
    try:
        if not os.path.exists(presets_path):
            tk.messagebox.showinfo("No Presets", "No saved presets found!")
            return
        
        with open(presets_path, 'r') as f:
            presets = json.load(f)
        
        if not presets:
            tk.messagebox.showinfo("No Presets", "No saved presets found!")
            return
        
        # Создаём окно выбора preset
        preset_window = tk.Toplevel(root)
        preset_window.title("Load Preset")
        preset_window.geometry("300x400")
        preset_window.configure(bg="#1a1a1a")
        preset_window.attributes("-topmost", True)
        
        tk.Label(preset_window, text="Select Preset:", bg="#1a1a1a",
                fg="white", font=("Arial", 10, "bold")).pack(pady=10)
        
        listbox = tk.Listbox(preset_window, bg="#333", fg="white",
                            selectmode=tk.SINGLE, font=("Arial", 9))
        listbox.pack(fill="both", expand=True, padx=10, pady=5)
        
        for preset_name in presets.keys():
            listbox.insert(tk.END, preset_name)
        
        def apply_preset():
            selection = listbox.curselection()
            if not selection:
                return
            
            preset_name = listbox.get(selection[0])
            preset_data = presets[preset_name]
            
            # Загружаем данные
            ent_x1.delete(0, tk.END)
            ent_x1.insert(0, preset_data.get("x1", "0"))
            ent_z1.delete(0, tk.END)
            ent_z1.insert(0, preset_data.get("z1", "0"))
            ent_x2.delete(0, tk.END)
            ent_x2.insert(0, preset_data.get("x2", "1"))
            ent_z2.delete(0, tk.END)
            ent_z2.insert(0, preset_data.get("z2", "1"))
            ent_y.delete(0, tk.END)
            ent_y.insert(0, preset_data.get("y1", "122.10214"))
            ent_y2.delete(0, tk.END)
            ent_y2.insert(0, preset_data.get("y2", "123.000"))
            ent_limit.delete(0, tk.END)
            ent_limit.insert(0, preset_data.get("limit", "1.000"))
            
            ent_x1_mm.delete(0, tk.END)
            ent_x1_mm.insert(0, preset_data.get("x1_mm", "0"))
            ent_z1_mm.delete(0, tk.END)
            ent_z1_mm.insert(0, preset_data.get("z1_mm", "0"))
            ent_x2_mm.delete(0, tk.END)
            ent_x2_mm.insert(0, preset_data.get("x2_mm", "1"))
            ent_z2_mm.delete(0, tk.END)
            ent_z2_mm.insert(0, preset_data.get("z2_mm", "1"))
            ent_mm_y.delete(0, tk.END)
            ent_mm_y.insert(0, preset_data.get("mm_y", "122.10214"))
            ent_mm_limit.delete(0, tk.END)
            ent_mm_limit.insert(0, preset_data.get("mm_limit", "1.000"))
            
            save_settings()
            preset_window.destroy()
        
        def delete_preset():
            selection = listbox.curselection()
            if not selection:
                return
            
            preset_name = listbox.get(selection[0])
            if tk.messagebox.askyesno("Delete Preset", f"Delete preset '{preset_name}'?"):
                del presets[preset_name]
                with open(presets_path, 'w') as f:
                    json.dump(presets, f, indent=2)
                listbox.delete(selection[0])
        
        btn_frame = tk.Frame(preset_window, bg="#1a1a1a")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Load", command=apply_preset,
                 bg="#4dff4d", font=("Arial", 9)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Delete", command=delete_preset,
                 bg="#ff4d4d", font=("Arial", 9)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=preset_window.destroy,
                 bg="#888", font=("Arial", 9)).pack(side="left", padx=5)
        
    except Exception as e:
        tk.messagebox.showerror("Error", f"Error loading presets: {e}")

# ================= GUI =================
root = tk.Tk()
root.title("MC Helper Ultimate")
root.geometry("400x750")
root.attributes("-topmost", True)
root.configure(bg="#1a1a1a")

# Canvas для скроллинга без видимой полоски
canvas = tk.Canvas(root, bg="#1a1a1a", highlightthickness=0)
scrollable_frame = tk.Frame(canvas, bg="#1a1a1a")

# Обновление scroll region при изменении размера
def update_scroll_region(event=None):
    canvas.configure(scrollregion=canvas.bbox("all"))
    # Делаем scrollable_frame такой же ширины как canvas
    canvas.itemconfig(canvas_window, width=event.width if event else canvas.winfo_width())

scrollable_frame.bind("<Configure>", update_scroll_region)
canvas.bind("<Configure>", update_scroll_region)

canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Скроллинг колёсиком без видимой полоски
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

root.bind_all("<MouseWheel>", _on_mousewheel)

canvas.pack(fill="both", expand=True)

label_coords = tk.Label(scrollable_frame, text="READY", fg="#00ff00",
                        bg="black", font=("Consolas", 12, "bold"), height=2)
label_coords.pack(pady=10, fill="x")

# --- UI SCALE SELECTOR (верхний левый угол) ---
ui_scale_frame = tk.Frame(scrollable_frame, bg="#1a1a1a")
ui_scale_frame.pack(anchor="w", padx=10, pady=(0, 5))

tk.Label(ui_scale_frame, text="UI Size:", fg="gray", bg="#1a1a1a", 
         font=("Arial", 8)).pack(side="left", padx=(0, 5))

def change_ui_scale(scale):
    """Смена размера интерфейса"""
    global ui_scale
    ui_scale = scale
    load_templates(scale)
    save_settings()
    print(f"[INFO] UI Scale changed to: {scale}")

ui_scale_var = tk.StringVar(value=ui_scale)

ui_scales = [
    ("Mini", "mini"),
    ("Default", "def"),
    ("Big", "big"),
    ("Auto", "auto")
]

for text, scale in ui_scales:
    tk.Radiobutton(ui_scale_frame, text=text, variable=ui_scale_var, value=scale,
                   command=lambda s=scale: change_ui_scale(s),
                   bg="#1a1a1a", fg="#00ff00", selectcolor="black",
                   font=("Arial", 7)).pack(side="left", padx=2)

# Кнопки presets в правом верхнем углу
preset_frame = tk.Frame(scrollable_frame, bg="#1a1a1a")
preset_frame.pack(anchor="e", padx=10)

tk.Button(preset_frame, text="Save Preset", command=save_preset,
         bg="#4d4dff", fg="white", font=("Arial", 8), bd=0).pack(side="left", padx=2)
tk.Button(preset_frame, text="Load Preset", command=load_preset,
         bg="#ff8c00", fg="white", font=("Arial", 8), bd=0).pack(side="left", padx=2)

# --- ЧЕКБОКСЫ ---
neo_var = tk.BooleanVar()
face_var = tk.BooleanVar()
show_coords_area = tk.BooleanVar()
show_facing_area = tk.BooleanVar()
show_offset_var = tk.BooleanVar()
show_x_offset_var = tk.BooleanVar()
show_z_offset_var = tk.BooleanVar()
show_mm_offset_var = tk.BooleanVar()
show_mm_x_offset_var = tk.BooleanVar()
show_mm_z_offset_var = tk.BooleanVar()

# Delta чекбоксы
use_delta_y1_var = tk.BooleanVar()
use_delta_y2_var = tk.BooleanVar()
use_delta_mm_var = tk.BooleanVar()

# Ignore чекбоксы
ignore_x_var = tk.BooleanVar()
ignore_z_var = tk.BooleanVar()
ignore_x_mm_var = tk.BooleanVar()
ignore_z_mm_var = tk.BooleanVar()

def toggle_neo_ui():
    if neo_var.get():
        label_y1.config(text="Target Height Y1 (For X):")
        neo_frame.pack(after=chk_neo, pady=5, fill="x")
    else:
        label_y1.config(text="Target Height Y1:")
        neo_frame.pack_forget()
    save_settings()

chk_neo = tk.Checkbutton(scrollable_frame, text="ENABLE Z-NEO",
                         variable=neo_var, command=toggle_neo_ui,
                         bg="#1a1a1a", fg="cyan", selectcolor="black")
chk_neo.pack()

tk.Checkbutton(scrollable_frame, text="ENABLE FACING",
               variable=face_var,
               bg="#1a1a1a", fg="yellow",
               selectcolor="black").pack()

# ---------- OVERLAY ----------
def make_scan_overlay(area, color):
    w = tk.Toplevel(root)
    w.overrideredirect(True)
    w.attributes("-topmost", True)
    w.attributes("-alpha", 0.35)
    w.configure(bg=color)
    w.withdraw()
    update_overlay(w, area)
    return w

def update_overlay(w, area):
    w.geometry(f"{area['width']}x{area['height']}+{area['left']}+{area['top']}")

coords_overlay = make_scan_overlay(MONITOR_COORDS, "#00ff00")
facing_overlay = make_scan_overlay(MONITOR_FACING, "#ffff00")

# ---------- РЕДАКТОР ЗОН ----------
def area_editor(area, overlay):
    frame = tk.Frame(scrollable_frame, bg="#1a1a1a")
    for k in ("top", "left", "width", "height"):
        row = tk.Frame(frame, bg="#1a1a1a")
        row.pack(fill="x", padx=40, pady=1)

        tk.Label(row, text=k.upper(), width=6,
                 fg="gray", bg="#1a1a1a").pack(side="left")

        e = tk.Entry(row, bg="#333", fg="white", bd=0, justify="center")
        e.insert(0, str(area[k]))
        e.pack(side="right", fill="x", expand=True)

        def bind(entry, key):
            def apply(*_):
                try:
                    area[key] = max(1, int(entry.get()))
                    update_overlay(overlay, area)
                    save_settings()  # Сохраняем при изменении
                except:
                    pass
            entry.bind("<KeyRelease>", apply)

        bind(e, k)
    return frame

coords_editor = area_editor(MONITOR_COORDS, coords_overlay)
facing_editor = area_editor(MONITOR_FACING, facing_overlay)

def toggle_coords_area():
    if show_coords_area.get():
        coords_overlay.deiconify()
        coords_editor.pack()
    else:
        coords_overlay.withdraw()
        coords_editor.pack_forget()
    save_settings()

def toggle_facing_area():
    if show_facing_area.get():
        facing_overlay.deiconify()
        facing_editor.pack()
    else:
        facing_overlay.withdraw()
        facing_editor.pack_forget()
    save_settings()

tk.Checkbutton(scrollable_frame, text="SHOW COORDS SCAN AREA",
               variable=show_coords_area,
               command=toggle_coords_area,
               bg="#1a1a1a", fg="#00ff00",
               selectcolor="black").pack()

tk.Checkbutton(scrollable_frame, text="SHOW FACING SCAN AREA",
               variable=show_facing_area,
               command=toggle_facing_area,
               bg="#1a1a1a", fg="yellow",
               selectcolor="black").pack()

# ---------- OFFSET CHECKBOXES ----------
def toggle_offset_window():
    if show_offset_var.get():
        # Применяем сохраненную геометрию если она есть
        if "offset_window_geometry" in settings_data:
            res_window.geometry(settings_data["offset_window_geometry"])
        res_window.deiconify()
    else:
        res_window.withdraw()
    save_settings()

def toggle_x_offset_window():
    if show_x_offset_var.get():
        # Применяем сохраненную геометрию если она есть
        if "x_offset_window_geometry" in settings_data:
            x_offset_window.geometry(settings_data["x_offset_window_geometry"])
        x_offset_window.deiconify()
    else:
        x_offset_window.withdraw()
    save_settings()

def toggle_z_offset_window():
    if show_z_offset_var.get():
        # Применяем сохраненную геометрию если она есть
        if "z_offset_window_geometry" in settings_data:
            z_offset_window.geometry(settings_data["z_offset_window_geometry"])
        z_offset_window.deiconify()
    else:
        z_offset_window.withdraw()
    save_settings()

def toggle_mm_offset_ui():
    # Показываем/скрываем поля MM
    if show_mm_offset_var.get() or show_mm_x_offset_var.get() or show_mm_z_offset_var.get():
        mm_offset_frame.pack(pady=5, fill="x")
    else:
        mm_offset_frame.pack_forget()
    
    # MM offset окно показывается только если включен SHOW MM OFFSET
    if show_mm_offset_var.get():
        # Применяем сохраненную геометрию если она есть
        if "mm_offset_window_geometry" in settings_data:
            mm_offset_window.geometry(settings_data["mm_offset_window_geometry"])
        mm_offset_window.deiconify()
    else:
        mm_offset_window.withdraw()
    save_settings()

def toggle_mm_x_offset_window():
    # Показываем поля если включен хотя бы один MM чекбокс
    if show_mm_offset_var.get() or show_mm_x_offset_var.get() or show_mm_z_offset_var.get():
        mm_offset_frame.pack(pady=5, fill="x")
    else:
        mm_offset_frame.pack_forget()
    
    if show_mm_x_offset_var.get():
        # Применяем сохраненную геометрию если она есть
        if "mm_x_offset_window_geometry" in settings_data:
            mm_x_offset_window.geometry(settings_data["mm_x_offset_window_geometry"])
        mm_x_offset_window.deiconify()
    else:
        mm_x_offset_window.withdraw()
    save_settings()

def toggle_mm_z_offset_window():
    # Показываем поля если включен хотя бы один MM чекбокс
    if show_mm_offset_var.get() or show_mm_x_offset_var.get() or show_mm_z_offset_var.get():
        mm_offset_frame.pack(pady=5, fill="x")
    else:
        mm_offset_frame.pack_forget()
    
    if show_mm_z_offset_var.get():
        # Применяем сохраненную геометрию если она есть
        if "mm_z_offset_window_geometry" in settings_data:
            mm_z_offset_window.geometry(settings_data["mm_z_offset_window_geometry"])
        mm_z_offset_window.deiconify()
    else:
        mm_z_offset_window.withdraw()
    save_settings()

tk.Checkbutton(scrollable_frame, text="SHOW OFFSET",
               variable=show_offset_var,
               command=toggle_offset_window,
               bg="#1a1a1a", fg="#4dff4d",
               selectcolor="black").pack()

tk.Checkbutton(scrollable_frame, text="  SHOW X OFFSET",
               variable=show_x_offset_var,
               command=toggle_x_offset_window,
               bg="#1a1a1a", fg="#4dff4d",
               selectcolor="black").pack()

tk.Checkbutton(scrollable_frame, text="  SHOW Z OFFSET",
               variable=show_z_offset_var,
               command=toggle_z_offset_window,
               bg="#1a1a1a", fg="#4dff4d",
               selectcolor="black").pack()

tk.Checkbutton(scrollable_frame, text="SHOW MM OFFSET",
               variable=show_mm_offset_var,
               command=toggle_mm_offset_ui,
               bg="#1a1a1a", fg="#ffa500",
               selectcolor="black").pack()

tk.Checkbutton(scrollable_frame, text="  SHOW X MM OFFSET",
               variable=show_mm_x_offset_var,
               command=toggle_mm_x_offset_window,
               bg="#1a1a1a", fg="#ffa500",
               selectcolor="black").pack()

tk.Checkbutton(scrollable_frame, text="  SHOW Z MM OFFSET",
               variable=show_mm_z_offset_var,
               command=toggle_mm_z_offset_window,
               bg="#1a1a1a", fg="#ffa500",
               selectcolor="black").pack()

# ---------- ПОЛЯ ВВОДА ----------
def field(parent, default):
    e = tk.Entry(parent, justify="center", bg="#333", fg="white", bd=0)
    e.insert(0, default)
    e.pack(pady=2, ipady=2, padx=50, fill="x")
    e.bind("<KeyRelease>", lambda event: save_settings())
    return e

tk.Label(scrollable_frame, text="Point 1 X:", fg="gray", bg="#1a1a1a", font=("Arial", 7)).pack()
tk.Checkbutton(scrollable_frame, text="Ignore X", variable=ignore_x_var,
               bg="#1a1a1a", fg="red", selectcolor="black",
               font=("Arial", 7), command=save_settings).pack(pady=(0, 2))
ent_x1 = field(scrollable_frame, "0")
tk.Label(scrollable_frame, text="Point 2 X:", fg="gray", bg="#1a1a1a", font=("Arial", 7)).pack()
ent_x2 = field(scrollable_frame, "1")

tk.Label(scrollable_frame, text="Point 1 Z:", fg="gray", bg="#1a1a1a", font=("Arial", 7)).pack()
tk.Checkbutton(scrollable_frame, text="Ignore Z", variable=ignore_z_var,
               bg="#1a1a1a", fg="red", selectcolor="black",
               font=("Arial", 7), command=save_settings).pack(pady=(0, 2))
ent_z1 = field(scrollable_frame, "0")
tk.Label(scrollable_frame, text="Point 2 Z:", fg="gray", bg="#1a1a1a", font=("Arial", 7)).pack()
ent_z2 = field(scrollable_frame, "1")

label_y1 = tk.Label(scrollable_frame, text="Target Height Y1:", fg="gray", bg="#1a1a1a", font=("Arial", 7))
label_y1.pack()
tk.Checkbutton(scrollable_frame, text="Use Delta", variable=use_delta_y1_var,
               bg="#1a1a1a", fg="cyan", selectcolor="black",
               font=("Arial", 7), command=save_settings).pack(pady=(0, 2))
ent_y = field(scrollable_frame, "122.10214")

neo_frame = tk.Frame(scrollable_frame, bg="#1a1a1a")
tk.Label(neo_frame, text="Target Height Y2 (For Z-Neo):", fg="gray", bg="#1a1a1a", font=("Arial", 7)).pack()
tk.Checkbutton(neo_frame, text="Use Delta (Auto 2-tick)", variable=use_delta_y2_var,
               bg="#1a1a1a", fg="cyan", selectcolor="black",
               font=("Arial", 7), command=save_settings).pack(pady=(0, 2))
ent_y2 = field(neo_frame, "123.000")

tk.Label(scrollable_frame, text="Show Limit:", fg="gray",
         bg="#1a1a1a", font=("Arial", 7)).pack()
ent_limit = field(scrollable_frame, "1.000")

# ---------- MM OFFSET ПОЛЯ ----------
mm_offset_frame = tk.Frame(scrollable_frame, bg="#1a1a1a")

tk.Label(mm_offset_frame, text="MM Point 1 X:", fg="orange", bg="#1a1a1a", font=("Arial", 7)).pack()
tk.Checkbutton(mm_offset_frame, text="Ignore X MM", variable=ignore_x_mm_var,
               bg="#1a1a1a", fg="red", selectcolor="black",
               font=("Arial", 7), command=save_settings).pack(pady=(0, 2))
ent_x1_mm = field(mm_offset_frame, "0")
tk.Label(mm_offset_frame, text="MM Point 2 X:", fg="orange", bg="#1a1a1a", font=("Arial", 7)).pack()
ent_x2_mm = field(mm_offset_frame, "1")

tk.Label(mm_offset_frame, text="MM Point 1 Z:", fg="orange", bg="#1a1a1a", font=("Arial", 7)).pack()
tk.Checkbutton(mm_offset_frame, text="Ignore Z MM", variable=ignore_z_mm_var,
               bg="#1a1a1a", fg="red", selectcolor="black",
               font=("Arial", 7), command=save_settings).pack(pady=(0, 2))
ent_z1_mm = field(mm_offset_frame, "0")
tk.Label(mm_offset_frame, text="MM Point 2 Z:", fg="orange", bg="#1a1a1a", font=("Arial", 7)).pack()
ent_z2_mm = field(mm_offset_frame, "1")

tk.Label(mm_offset_frame, text="MM Target Height Y:", fg="orange", bg="#1a1a1a", font=("Arial", 7)).pack()
tk.Checkbutton(mm_offset_frame, text="Use Delta", variable=use_delta_mm_var,
               bg="#1a1a1a", fg="cyan", selectcolor="black",
               font=("Arial", 7), command=save_settings).pack(pady=(0, 2))
ent_mm_y = field(mm_offset_frame, "122.10214")

tk.Label(mm_offset_frame, text="MM Show Limit:", fg="orange", bg="#1a1a1a", font=("Arial", 7)).pack()
ent_mm_limit = field(mm_offset_frame, "1.000")

tk.Label(scrollable_frame, text="Facing HEX Color:",
         fg="yellow", bg="#1a1a1a", font=("Arial", 7)).pack()
ent_face_color = field(scrollable_frame, "#FFFF00")

def toggle():
    global is_running
    is_running = not is_running
    btn.config(text="STOP" if is_running else "START",
               bg="#ff4d4d" if is_running else "#4dff4d")
    if is_running:
        threading.Thread(target=scan_logic, daemon=True).start()

btn = tk.Button(scrollable_frame, text="START", command=toggle,
                bg="#4dff4d", font=("Arial", 10, "bold"))
btn.pack(pady=15)

# ---------- OVERLAY ОКНА ----------
def make_overlay(g, start_text, color):
    w = tk.Toplevel(root)
    w.overrideredirect(True)
    w.attributes("-topmost", True)
    w.geometry(g)
    w.configure(bg="black")
    w.attributes("-transparentcolor", "black")

    l = tk.Label(
        w,
        text=start_text,
        fg=color,
        bg="black",
        font=("Consolas", 24, "bold")
    )
    l.pack(expand=True, fill="both")

    # --- ПЕРЕТАСКИВАНИЕ ---
    def start_move(e):
        w.x = e.x
        w.y = e.y

    def do_move(e):
        w.geometry(f"+{w.winfo_x() + (e.x - w.x)}+{w.winfo_y() + (e.y - w.y)}")

    l.bind("<Button-1>", start_move)
    l.bind("<B1-Motion>", do_move)

    # --- РЕСАЙЗ ---
    grip = tk.Label(w, text="◢", fg="#111", bg="black", cursor="size_nw_se")
    grip.place(relx=1.0, rely=1.0, anchor="se")

    def start_resize(e):
        w.w = w.winfo_width()
        w.h = w.winfo_height()
        w.sx = e.x_root
        w.sy = e.y_root

    def do_resize(e):
        nw = w.w + (e.x_root - w.sx)
        nh = w.h + (e.y_root - w.sy)
        if nw > 40 and nh > 20:
            w.geometry(f"{nw}x{nh}")
            l.config(font=("Consolas", int(nh / 1.8), "bold"))

    grip.bind("<Button-1>", start_resize)
    grip.bind("<B1-Motion>", do_resize)

    return w, l

# Создание всех окон
face_window, face_label = make_overlay("120x50+200+200", "0.0", "#FFFF00")
res_window, res_label = make_overlay("280x60+200+300", "WAIT", "#4dff4d")
x_offset_window, x_offset_label = make_overlay("280x60+500+300", "X: WAIT", "#4dff4d")
z_offset_window, z_offset_label = make_overlay("280x60+800+300", "Z: WAIT", "#4dff4d")
mm_offset_window, mm_offset_label = make_overlay("280x60+200+400", "MM: WAIT", "#ffa500")
mm_x_offset_window, mm_x_offset_label = make_overlay("280x60+500+400", "MM X: WAIT", "#ffa500")
mm_z_offset_window, mm_z_offset_label = make_overlay("280x60+800+400", "MM Z: WAIT", "#ffa500")

# Изначально скрыты все окна
face_window.withdraw()
res_window.withdraw()
x_offset_window.withdraw()
z_offset_window.withdraw()
mm_offset_window.withdraw()
mm_x_offset_window.withdraw()
mm_z_offset_window.withdraw()

def show_res(v):
    if show_offset_var.get():
        if abs(v) < 0.0005:  # Если ровно 0.000 (с округлением)
            res_label.config(
                text=f"OFFSET: ±{abs(v):.3f}",
                fg="#ffa500")  # Оранжевый
        else:
            res_label.config(
                text=f"OFFSET: {'+' if v >= 0 else ''}{v:.3f}",
                fg="#4dff4d" if v >= 0 else "#ff4d4d")
        res_window.deiconify()

def show_x_offset(v):
    if show_x_offset_var.get():
        if abs(v) < 0.0005:  # Если ровно 0.000
            x_offset_label.config(
                text=f"X: ±{abs(v):.3f}",
                fg="#ffa500")  # Оранжевый
        else:
            x_offset_label.config(
                text=f"X: {'+' if v >= 0 else ''}{v:.3f}",
                fg="#4dff4d" if v >= 0 else "#ff4d4d")
        x_offset_window.deiconify()

def show_z_offset(v):
    if show_z_offset_var.get():
        if abs(v) < 0.0005:  # Если ровно 0.000
            z_offset_label.config(
                text=f"Z: ±{abs(v):.3f}",
                fg="#ffa500")  # Оранжевый
        else:
            z_offset_label.config(
                text=f"Z: {'+' if v >= 0 else ''}{v:.3f}",
                fg="#4dff4d" if v >= 0 else "#ff4d4d")
        z_offset_window.deiconify()

def show_mm_res(v):
    if show_mm_offset_var.get():
        if abs(v) < 0.0005:  # Если ровно 0.000
            mm_offset_label.config(
                text=f"MM: ±{abs(v):.3f}",
                fg="#ffa500")  # Оранжевый
        else:
            mm_offset_label.config(
                text=f"MM: {'+' if v >= 0 else ''}{v:.3f}",
                fg="#4dff4d" if v >= 0 else "#ff4d4d")
        mm_offset_window.deiconify()

def show_mm_x_offset(v):
    if show_mm_x_offset_var.get():
        if abs(v) < 0.0005:  # Если ровно 0.000
            mm_x_offset_label.config(
                text=f"MM X: ±{abs(v):.3f}",
                fg="#ffa500")  # Оранжевый
        else:
            mm_x_offset_label.config(
                text=f"MM X: {'+' if v >= 0 else ''}{v:.3f}",
                fg="#4dff4d" if v >= 0 else "#ff4d4d")
        mm_x_offset_window.deiconify()

def show_mm_z_offset(v):
    if show_mm_z_offset_var.get():
        if abs(v) < 0.0005:  # Если ровно 0.000
            mm_z_offset_label.config(
                text=f"MM Z: ±{abs(v):.3f}",
                fg="#ffa500")  # Оранжевый
        else:
            mm_z_offset_label.config(
                text=f"MM Z: {'+' if v >= 0 else ''}{v:.3f}",
                fg="#4dff4d" if v >= 0 else "#ff4d4d")
        mm_z_offset_window.deiconify()

# Загружаем настройки при старте
root.after(100, load_settings)

# Сохраняем настройки при закрытии
root.protocol("WM_DELETE_WINDOW", lambda: (save_settings(), root.destroy()))

update_calc()
root.mainloop()
