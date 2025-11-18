#copyright joshuah.rainstar@gmail.com
#note: this WILL not look very good with large batches.
#recommend you slice your batch,logits to feed only a few thousand letters.
#if your model completes in less than a second per batch, this will slow it down.

import numpy as np
import torch
import torch.nn.functional as F
import ipywidgets as widgets
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
import io
import math
import time

class MatrixDashboard:
    def __init__(self, batch_size, seq_len, itos=None):
        self.target_cells = batch_size * seq_len
        self.itos_map = itos if itos is not None else {}
        
        # --- 1. Geometry & Font Setup ---
        # Cinematic aspect ratio logic (approx 2.5:1)
        self.rows = int(math.sqrt(self.target_cells / 5))
        self.cols = int(np.ceil(self.target_cells / self.rows))
        self.n_cells = self.rows * self.cols
        
        # Visual constants
        self.cell_w = 10  # pixel width per char
        self.cell_h = 16  # pixel height per char
        self.width = self.cols * self.cell_w
        self.height = self.rows * self.cell_h + 40 # +40 for stats bar
        
        # Load Font (Robust Fallback)
        try:
            self.font = ImageFont.truetype("DejaVuSansMono.ttf", 11)
        except:
            try:
                self.font = ImageFont.truetype("Courier New.ttf", 11)
            except:
                self.font = ImageFont.load_default()

        # --- 2. Decoder ---
        if itos is not None:
            def safe_decode(x):
                c = itos.get(x, "?")
                if c == "\n": return "¶"
                if c == "\t": return "→"
                if c == " ": return "·"
                return c
            self.decode = safe_decode
        else:
            self.decode = lambda x: chr(x) if 32 <= x <= 126 else "?"

        # --- 3. Simulation State ---
        self.display_chars = ["·"] * self.n_cells
        self.display_colors = [(40, 40, 40)] * self.n_cells 
        self.freshness = np.zeros(self.n_cells, dtype=np.float32)
        self.ewma_loss = None
        self.step = 0
        
        # --- 4. Widget Setup ---
        self.out_widget = widgets.Image(format='png', width=self.width, height=self.height)
        self.layout = widgets.VBox([self.out_widget])

    def render(self):
        """Display the widget in the notebook."""
        display(self.layout)

    def update(self, yb, logits, loss_val):
        """
        Update grid. 
        Logic: 
        - If Predicted Token is NOT in itos, display Target Token.
        - Colors: Green (Correct), Orange (Incorrect).
        """
        self.step += 1
        
        # --- 1. Tensor Ops ---
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            p_max, preds = torch.max(probs, dim=-1)
            
            p_max = p_max.cpu().numpy().flatten()
            preds = preds.cpu().numpy().flatten()
            targets = yb.cpu().numpy().flatten()

        # Limit to grid size
        limit = min(len(p_max), self.n_cells)
        
        # --- 2. Life/Freshness Simulation ---
        is_correct = (preds[:limit] == targets[:limit]).astype(np.float32)
        self.freshness *= 0.92 # Decay global freshness
        
        # Update Rule: Update if New Confidence > Old Freshness OR Old Freshness < 0.10 (faded)
        current_freshness = self.freshness[:limit]
        update_mask = (p_max[:limit] > current_freshness) | (current_freshness < 0.10)
        
        # Apply updates to freshness buffer
        self.freshness[:limit] = np.where(update_mask, p_max[:limit], current_freshness)
        
        # --- 3. Color Calculation (Vectorized) ---
        # Only calculate for updated cells
        update_indices = np.where(update_mask)[0]
        
        if len(update_indices) > 0:
            # Get subset of values
            vals = p_max[:limit][update_indices] * 255.0
            vals = np.maximum(50.0, vals) # Minimum brightness so nothing is invisible
            corrects = is_correct[update_indices]
            
            # RGB Logic
            # Correct (Greenish): R=0.5v, G=1.0v, B=0.25v
            # Incorrect (Orange): R=1.0v, G=0.5v, B=0.0v
            r = (corrects * (vals * 0.5) + (1 - corrects) * vals).astype(np.int32)
            g = (corrects * vals + (1 - corrects) * (vals * 0.5)).astype(np.int32)
            b = (corrects * (vals * 0.25)).astype(np.int32)
            
            # --- 4. Update State Lists (The Loop) ---
            # We iterate only the changed indices
            for i, idx in enumerate(update_indices):
                token_id = preds[idx]
                target_id = targets[idx]
                
                # PATCH: Fallback to Target if Prediction is OOV
                if self.itos_map and (token_id not in self.itos_map):
                    token_id = target_id
                
                self.display_chars[idx] = self.decode(token_id)
                self.display_colors[idx] = (r[i], g[i], b[i])

        # --- 5. Rendering (PIL) ---
        img = Image.new("RGB", (self.width, self.height), (10, 10, 10))
        draw = ImageDraw.Draw(img)
        
        # Optimization: Local variable references for loop speed
        d_text = draw.text
        fnt = self.font
        cw, ch = self.cell_w, self.cell_h
        cols = self.cols
        chars = self.display_chars
        colors = self.display_colors
        
        for i in range(self.n_cells):
            y_row = i // cols
            x_col = i % cols
            
            px = x_col * cw
            py = y_row * ch + 40 # Offset for stats bar
            
            d_text((px, py), chars[i], font=fnt, fill=colors[i])

        # --- 6. Stats Bar ---
        if self.ewma_loss is None: self.ewma_loss = loss_val
        else: self.ewma_loss = 0.95 * self.ewma_loss + 0.05 * loss_val
        
        acc = np.mean(is_correct)
        
        # Stats Background
        draw.rectangle([0, 0, self.width, 35], fill=(20, 20, 20))
        
        # Stats Text
        draw.text((10, 10), f"STEP: {self.step}", font=fnt, fill=(200, 200, 200))
        draw.text((100, 10), f"LOSS: {loss_val:.4f}", font=fnt, fill=(255, 100, 100))
        draw.text((220, 10), f"EWMA: {self.ewma_loss:.4f}", font=fnt, fill=(255, 255, 0))
        draw.text((340, 10), f"ACC: {acc:.1%}", font=fnt, fill=(0, 255, 0))

        # --- 7. Push to Widget ---
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            self.out_widget.value = output.getvalue()

# Usage
#dashboard = MatrixDashboard(batch_size, block_size, itos=itos)
#dashboard.render()
#dashboard.update(yb, logits, loss.item())
