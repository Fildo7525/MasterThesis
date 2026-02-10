import tkinter as tk
from PIL import Image, ImageTk

gif_path = "/home/samuel/test/MasterThesis/Orthomosaics/translated/mid/translated_8x_8y/processed_output/tiles_with_labels.gif"

class GIFPlayer:
    def __init__(self, root, gif_path):
        self.root = root
        self.root.title("GIF Player")
        
        # Load GIF
        self.gif = Image.open(gif_path)
        self.frames = []
        try:
            while True:
                self.frames.append(self.gif.copy())
                self.gif.seek(self.gif.tell() + 1)
        except EOFError:
            pass
        
        self.current_frame = 0
        self.is_playing = True
        
        # Create canvas
        self.canvas = tk.Canvas(root, width=800, height=800)
        self.canvas.pack()
        
        # Display first frame
        self.photo = ImageTk.PhotoImage(self.frames[0])
        self.image_on_canvas = self.canvas.create_image(400, 400, image=self.photo)
        
        # Frame counter
        self.frame_label = tk.Label(root, text=f"Frame: 1/{len(self.frames)}")
        self.frame_label.pack()
        
        # Control buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        self.btn_rewind = tk.Button(button_frame, text="⏮ Rewind", command=self.rewind)
        self.btn_rewind.pack(side=tk.LEFT, padx=5)
        
        self.btn_prev = tk.Button(button_frame, text="◀ Prev", command=self.prev_frame)
        self.btn_prev.pack(side=tk.LEFT, padx=5)
        
        self.btn_play = tk.Button(button_frame, text="⏸ Pause", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_next = tk.Button(button_frame, text="Next ▶", command=self.next_frame)
        self.btn_next.pack(side=tk.LEFT, padx=5)
        
        # Start animation
        self.animate()
    
    def animate(self):
        if self.is_playing:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.update_frame()
        self.root.after(1000, self.animate)  # 1000ms delay
    
    def update_frame(self):
        self.photo = ImageTk.PhotoImage(self.frames[self.current_frame])
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo)
        self.frame_label.config(text=f"Frame: {self.current_frame + 1}/{len(self.frames)}")
    
    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.config(text="⏸ Pause" if self.is_playing else "▶ Play")
    
    def rewind(self):
        self.current_frame = 0
        self.update_frame()
        self.is_playing = False
        self.btn_play.config(text="▶ Play")
    
    def prev_frame(self):
        self.is_playing = False
        self.btn_play.config(text="▶ Play")
        self.current_frame = (self.current_frame - 1) % len(self.frames)
        self.update_frame()
    
    def next_frame(self):
        self.is_playing = False
        self.btn_play.config(text="▶ Play")
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.update_frame()

# Run the player
root = tk.Tk()
player = GIFPlayer(root, gif_path)
root.mainloop()