#!/usr/bin/env python3
"""
Standalone Map Visualizer: Multi-Line & Circular Patch Finder
-----------------------------------------------------------
Features:
1. Linear Analysis: Finds longest straight chains with Angle & Distance exclusions.
2. Circular Analysis: Finds largest open areas with Overlap Tolerance.
3. Interactive Sliders with Real-Time Value Displays.
"""

import struct
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import numpy as np
import math
import sys

# --- Matplotlib imports ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm 

# Increase recursion depth
sys.setrecursionlimit(5000)

# ==============================================================================
# --- Data Structures ---
# ==============================================================================

class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Tri:
    def __init__(self, v_indices, n_indices, data):
        self.vertex_indices = v_indices
        self.neighbour_indices = n_indices
        self.passable = data & 0b01111111 

# ==============================================================================
# --- Parser ---
# ==============================================================================

def parse_map_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            header_data = struct.unpack_from('<4s 12x I I I 7I I', content, 0)
            if header_data[0] != b'MAP1': return None, None
            
            navmesh_base = header_data[3]
            nav_data = struct.unpack_from('<10x H f 8x I I', content, navmesh_base)
            
            v_count, scale = nav_data[0], nav_data[1]
            v_ptr, t_info_ptr = nav_data[2], nav_data[3]

            # Vertices
            vertices = []
            v_fmt = '<hhh2x'
            v_size = struct.calcsize(v_fmt)
            v_addr = navmesh_base + v_ptr
            for i in range(v_count):
                d = struct.unpack_from(v_fmt, content, v_addr + i * v_size)
                vertices.append(Vertex(10*d[0]/scale, 10*d[1]/scale, 10*d[2]/scale))

            # Triangles
            t_info = struct.unpack_from('<8x I I', content, navmesh_base + t_info_ptr)
            t_count, t_ptr = t_info[0], t_info[1]
            tris = []
            t_fmt = '<3H 3h I'
            t_size = struct.calcsize(t_fmt)
            t_addr = navmesh_base + t_ptr
            for i in range(t_count):
                d = struct.unpack_from(t_fmt, content, t_addr + i * t_size)
                tris.append(Tri(d[0:3], d[3:6], d[6]))
            
            return vertices, tris
    except Exception as e:
        print(f"Parse Error: {e}")
        return None, None

# ==============================================================================
# --- Linear Logic (Straight Lines) ---
# ==============================================================================

def build_physical_edge_graph(tris, vertex_count):
    adj = {i: set() for i in range(vertex_count)}
    for tri in tris:
        if tri.passable != 0: continue
        v = tri.vertex_indices
        adj[v[0]].add(v[1]); adj[v[1]].add(v[0])
        adj[v[1]].add(v[2]); adj[v[2]].add(v[1])
        adj[v[2]].add(v[0]); adj[v[0]].add(v[2])
    return {k: list(v) for k, v in adj.items()}

def calculate_deflection(p1, p2, p3):
    v1 = (p2.x - p1.x, p2.z - p1.z)
    v2 = (p3.x - p2.x, p3.z - p2.z)
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    diff = math.degrees(abs(a1 - a2))
    if diff > 180: diff = 360 - diff
    return diff

def get_global_angle_of_path(vertices, path_indices):
    if len(path_indices) < 2: return 0.0
    start, end = vertices[path_indices[0]], vertices[path_indices[-1]]
    return math.degrees(math.atan2(end.z - start.z, end.x - start.x))

def get_path_centroid(vertices, path_indices):
    """Calculates the center point (average x, z) of a path."""
    if not path_indices: return (0,0)
    sum_x = sum(vertices[i].x for i in path_indices)
    sum_z = sum(vertices[i].z for i in path_indices)
    count = len(path_indices)
    return (sum_x / count, sum_z / count)

def is_angle_similar(new_angle, existing_angles, threshold):
    for ex in existing_angles:
        diff1 = min(abs(new_angle - ex), 360 - abs(new_angle - ex))
        diff2 = min(abs(new_angle - (ex+180)%360), 360 - abs(new_angle - (ex+180)%360))
        if diff1 < threshold or diff2 < threshold: return True
    return False

def find_multiple_lines(vertices, tris, count, angle_exclusion, min_separation):
    adj_graph = build_physical_edge_graph(tris, len(vertices))
    memo = {}
    visiting = set()

    def dfs_chain(prev, curr):
        state = (prev, curr)
        if state in memo: return memo[state]
        if state in visiting:
            d = math.hypot(vertices[curr].x - vertices[prev].x, vertices[curr].z - vertices[prev].z)
            return (d, [prev, curr])
        
        visiting.add(state)
        curr_v, prev_v = vertices[curr], vertices[prev]
        base_dist = math.hypot(curr_v.x - prev_v.x, curr_v.z - prev_v.z)
        
        best_len, best_path = 0.0, []
        
        if curr in adj_graph:
            for nxt in adj_graph[curr]:
                if nxt == prev: continue
                if calculate_deflection(prev_v, curr_v, vertices[nxt]) <= 15.0:
                    t_len, t_path = dfs_chain(curr, nxt)
                    if t_len > best_len:
                        best_len, best_path = t_len, t_path
        
        visiting.remove(state)
        total = base_dist + best_len
        path = [prev] + best_path[1:] if best_path else [prev, curr]
        memo[state] = (total, path)
        return (total, path)

    candidates = []
    for u in adj_graph:
        for v in adj_graph[u]:
            ln, p = dfs_chain(u, v)
            if ln > 5.0: candidates.append((ln, p))

    candidates.sort(key=lambda x: x[0], reverse=True)
    
    final_results = []
    excluded_angles = []
    
    for _, path in candidates:
        if len(final_results) >= count: break
        
        # 1. Angle Check
        angle = get_global_angle_of_path(vertices, path)
        if is_angle_similar(angle, excluded_angles, angle_exclusion): continue
        
        # 2. Separation Check (Distance)
        if min_separation > 0:
            cx, cz = get_path_centroid(vertices, path)
            too_close = False
            for existing_path in final_results:
                ex_cx, ex_cz = get_path_centroid(vertices, existing_path)
                dist = math.hypot(cx - ex_cx, cz - ex_cz)
                if dist < min_separation:
                    too_close = True
                    break
            if too_close: continue

        final_results.append(path)
        excluded_angles.append(angle)

    return final_results

# ==============================================================================
# --- Circular Logic (Open Areas) ---
# ==============================================================================

def find_circular_patches(vertices, tris, min_radius, max_count, overlap_tolerance):
    """
    overlap_tolerance: 0.0 (No overlap) to 1.0 (Full overlap allowed)
    """
    edge_to_tris = {}
    passable_tris = [t for t in tris if t.passable == 0]
    
    for t_idx, tri in enumerate(passable_tris):
        vs = tri.vertex_indices
        edges = [tuple(sorted((vs[0], vs[1]))), 
                 tuple(sorted((vs[1], vs[2]))), 
                 tuple(sorted((vs[2], vs[0])))]
        for e in edges:
            if e not in edge_to_tris: edge_to_tris[e] = []
            edge_to_tris[e].append(t_idx)

    boundary_edges = [e for e, t_list in edge_to_tris.items() if len(t_list) == 1]
    if not boundary_edges: return []

    wall_points = []
    for e in boundary_edges:
        v1, v2 = vertices[e[0]], vertices[e[1]]
        wall_points.append((v1.x, v1.z))
        wall_points.append((v2.x, v2.z))
        wall_points.append(((v1.x+v2.x)/2, (v1.z+v2.z)/2))
    
    wall_points_np = np.array(wall_points)
    candidates = []
    for tri in passable_tris:
        v0, v1, v2 = vertices[tri.vertex_indices[0]], vertices[tri.vertex_indices[1]], vertices[tri.vertex_indices[2]]
        cx, cy, cz = (v0.x+v1.x+v2.x)/3, (v0.y+v1.y+v2.y)/3, (v0.z+v1.z+v2.z)/3
        candidates.append({'pos': (cx, cy, cz), 'r': 0.0})

    if not candidates: return []

    cand_pos_np = np.array([(c['pos'][0], c['pos'][2]) for c in candidates])
    
    chunk_size = 100
    for i in range(0, len(candidates), chunk_size):
        chunk_cands = cand_pos_np[i:i+chunk_size]
        diff = chunk_cands[:, np.newaxis, :] - wall_points_np[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        min_dists = np.sqrt(np.min(dists_sq, axis=1))
        for j, val in enumerate(min_dists):
            candidates[i+j]['r'] = val

    candidates.sort(key=lambda x: x['r'], reverse=True)
    final_circles = []
    
    # Strictness logic: 
    # If tolerance is 1.0, strictness is 0 (overlap allowed)
    # If tolerance is 0.0, strictness is 1 (distance must be sum of radii)
    strictness = 1.0 - overlap_tolerance

    for cand in candidates:
        if len(final_circles) >= max_count: break
        if cand['r'] < min_radius: continue
        
        overlap = False
        for exist in final_circles:
            dist = math.hypot(cand['pos'][0] - exist['pos'][0], cand['pos'][2] - exist['pos'][2])
            
            # Minimum required distance based on tolerance
            required_dist = (exist['r'] + cand['r']) * strictness
            
            if dist < required_dist:
                overlap = True
                break
        
        if not overlap:
            final_circles.append(cand)
            
    return final_circles

# ==============================================================================
# --- UI & Visualization ---
# ==============================================================================

class MapVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Map Analysis Tool")
        self.root.geometry("1300x900")
        
        self.vertices = []
        self.tris = []
        
        # Main Layout
        self.pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.pane.pack(fill=tk.BOTH, expand=True)
        
        self.left_frame = ttk.Frame(self.pane)
        self.right_frame = ttk.Frame(self.pane, padding=10)
        self.pane.add(self.left_frame, stretch="always")
        self.pane.add(self.right_frame, minsize=300)

        # --- Sidebar ---
        
        ttk.Label(self.right_frame, text="Analysis Config", font=("Arial", 14, "bold")).pack(pady=(0,10))
        
        # --- Section 1: Lines ---
        lab_frame_lines = ttk.LabelFrame(self.right_frame, text="Linear Analysis (Lines)", padding=10)
        lab_frame_lines.pack(fill=tk.X, pady=5)
        
        # Slider 1: Line Count
        self.var_ln_count = tk.IntVar(value=10)
        self.create_labeled_slider(lab_frame_lines, "Line Count", self.var_ln_count, 0, 50, is_int=True)
        
        # Slider 2: Angle Exclusion
        self.var_ln_angle = tk.DoubleVar(value=15.0)
        self.create_labeled_slider(lab_frame_lines, "Angle Exclusion (°)", self.var_ln_angle, 0, 90)

        # Slider 3: Line Separation (New)
        self.var_ln_sep = tk.DoubleVar(value=10.0)
        self.create_labeled_slider(lab_frame_lines, "Min Separation (Dist)", self.var_ln_sep, 0, 100)

        # --- Section 2: Circles ---
        lab_frame_circ = ttk.LabelFrame(self.right_frame, text="Circular Analysis (Patches)", padding=10)
        lab_frame_circ.pack(fill=tk.X, pady=10)
        
        # Slider 4: Patch Count
        self.var_circ_count = tk.IntVar(value=5)
        self.create_labeled_slider(lab_frame_circ, "Patch Count", self.var_circ_count, 0, 20, is_int=True)
        
        # Slider 5: Min Radius
        self.var_circ_rad = tk.DoubleVar(value=5.0)
        self.create_labeled_slider(lab_frame_circ, "Min Radius", self.var_circ_rad, 1.0, 50.0)

        # Slider 6: Overlap Tolerance (New)
        self.var_circ_tol = tk.DoubleVar(value=0.2) # Default 20% overlap
        self.create_labeled_slider(lab_frame_circ, "Overlap Tolerance", self.var_circ_tol, 0.0, 1.0)

        # --- Controls ---
        self.btn_update = ttk.Button(self.right_frame, text="Update / Recalculate", command=self.run_analysis, state=tk.DISABLED)
        self.btn_update.pack(fill=tk.X, ipady=8, pady=20)

        self.status_var = tk.StringVar(value="Load a map to begin.")
        ttk.Label(self.right_frame, textvariable=self.status_var, wraplength=250, foreground="blue").pack(anchor="center")

        # Menu
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open .vpa Map...", command=self.open_map)
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        root.config(menu=menubar)

        # Matplotlib
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.left_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_labeled_slider(self, parent, label_text, variable, min_val, max_val, is_int=False):
        """Helper to create consistent slider UI"""
        lbl = ttk.Label(parent, text=f"{label_text}: {variable.get():.2f}")
        lbl.pack(anchor="w", pady=(5,0))
        
        def update_label(v):
            val = float(v)
            if is_int: val = int(val)
            lbl.config(text=f"{label_text}: {val:.2f}")

        scale = ttk.Scale(
            parent, from_=min_val, to=max_val, variable=variable,
            command=update_label
        )
        scale.pack(fill=tk.X)
        return scale

    def open_map(self):
        path = filedialog.askopenfilename(filetypes=[("VPA Map", "*.vpa")])
        if path:
            self.status_var.set("Parsing...")
            self.root.update()
            v, t = parse_map_file(path)
            if v and t:
                self.vertices = v
                self.tris = t
                self.btn_update.config(state=tk.NORMAL)
                self.run_analysis()
            else:
                self.status_var.set("Error parsing file.")

    def run_analysis(self):
        if not self.vertices: return
        self.btn_update.config(state=tk.DISABLED)
        
        try:
            # 1. Lines
            ln_c = self.var_ln_count.get()
            ln_ang = self.var_ln_angle.get()
            ln_sep = self.var_ln_sep.get() # New
            
            lines = []
            if ln_c > 0:
                self.status_var.set("Calculating Lines...")
                self.root.update()
                lines = find_multiple_lines(self.vertices, self.tris, ln_c, ln_ang, ln_sep)

            # 2. Circles
            cc_c = self.var_circ_count.get()
            cc_rad = self.var_circ_rad.get()
            cc_tol = self.var_circ_tol.get() # New
            
            circles = []
            if cc_c > 0:
                self.status_var.set("Calculating Circular Patches...")
                self.root.update()
                circles = find_circular_patches(self.vertices, self.tris, cc_rad, cc_c, cc_tol)
            
            self.draw_results(lines, circles)
            self.status_var.set(f"Done. {len(lines)} lines, {len(circles)} patches.")
            
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            print(e)
        
        self.btn_update.config(state=tk.NORMAL)

    def draw_results(self, lines, circles):
        self.ax.clear()
        
        # Draw Mesh
        polys = []
        colors = []
        for tri in self.tris:
            if tri.passable != 0: continue
            pts = [(self.vertices[i].x, self.vertices[i].y, self.vertices[i].z) for i in tri.vertex_indices]
            polys.append(pts)
            colors.append('lightgray')

        if polys:
            mesh_col = Poly3DCollection(polys, facecolors=colors, edgecolors='black', linewidths=0.1, alpha=0.1)
            self.ax.add_collection3d(mesh_col)
        
        # Draw Lines (Red->Blue)
        cmap = cm.get_cmap('jet_r')
        for i, path in enumerate(lines):
            color = cmap(i / max(1, len(lines)))
            xs = [self.vertices[p].x for p in path]
            ys = [self.vertices[p].y for p in path]
            zs = [self.vertices[p].z for p in path]
            self.ax.plot(xs, ys, zs, color=color, linewidth=3)
            self.ax.scatter([xs[0]], [ys[0]], [zs[0]], color=color, s=20)

        # Draw Circles (Cyan)
        theta = np.linspace(0, 2*np.pi, 50)
        for circ in circles:
            cx, cy, cz = circ['pos']
            r = circ['r']
            
            # Generate circle points on XZ plane
            x_c = cx + r * np.cos(theta)
            z_c = cz + r * np.sin(theta)
            y_c = np.full_like(x_c, cy + 2.0) # Lift slightly above mesh
            
            self.ax.plot(x_c, y_c, z_c, color='cyan', linewidth=2, linestyle='--')
            self.ax.scatter([cx], [cy], [cz], color='cyan', marker='x')

        # Scale
        all_x = [v.x for v in self.vertices]
        all_y = [v.y for v in self.vertices]
        all_z = [v.z for v in self.vertices]
        if all_x:
            mx = max(max(all_x)-min(all_x), max(all_z)-min(all_z)) / 2
            cx, cy, cz = np.mean(all_x), np.mean(all_y), np.mean(all_z)
            self.ax.set_xlim(cx - mx, cx + mx)
            self.ax.set_ylim(cy - mx, cy + mx)
            self.ax.set_zlim(cz - mx, cz + mx)
        
        self.ax.set_title(f"Analysis Result")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = MapVisualizerApp(root)
    root.mainloop()
