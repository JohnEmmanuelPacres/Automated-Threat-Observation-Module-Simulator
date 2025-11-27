import customtkinter as ctk

class ATMKeypad(ctk.CTkFrame):
    def __init__(self, master, on_key_press=None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_key_press = on_key_press
        
        # Configure grid layout
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)
        
        # Button configuration
        btn_opts = {
            "width": 80, 
            "height": 80, 
            "font": ("Arial", 24, "bold"),
            "corner_radius": 10
        }
        
        # Number keys 1-9
        keys = [
            ('1', 0, 0), ('2', 0, 1), ('3', 0, 2),
            ('4', 1, 0), ('5', 1, 1), ('6', 1, 2),
            ('7', 2, 0), ('8', 2, 1), ('9', 2, 2),
            ('0', 3, 1)
        ]
        
        for key, row, col in keys:
            btn = ctk.CTkButton(
                self, 
                text=key, 
                fg_color="#E0E0E0", 
                text_color="black",
                hover_color="#D0D0D0",
                command=lambda k=key: self._handle_press(k),
                **btn_opts
            )
            btn.grid(row=row, column=col, padx=5, pady=5)
            
        # Function keys
        # CANCEL (Red)
        self._create_func_btn("CANCEL", "red", 0, 3, btn_opts)
        # CLEAR (Yellow)
        self._create_func_btn("CLEAR", "#FFD700", 1, 3, btn_opts, text_color="black")
        # ENTER (Green)
        self._create_func_btn("ENTER", "green", 2, 3, btn_opts)
        
        # Blank keys (visual placeholders)
        self._create_blank_btn(3, 0, btn_opts)
        self._create_blank_btn(3, 2, btn_opts)
        self._create_blank_btn(3, 3, btn_opts)

    def _create_func_btn(self, text, color, row, col, opts, text_color="white"):
        func_opts = opts.copy()
        func_opts["font"] = ("Arial", 14, "bold")
        btn = ctk.CTkButton(
            self,
            text=text,
            fg_color=color,
            text_color=text_color,
            hover_color=color,
            command=lambda k=text: self._handle_press(k),
            **func_opts
        )
        btn.grid(row=row, column=col, padx=5, pady=5)

    def _create_blank_btn(self, row, col, opts):
        btn = ctk.CTkButton(
            self,
            text="",
            fg_color="transparent",
            state="disabled",
            **opts
        )
        btn.grid(row=row, column=col, padx=5, pady=5)

    def _handle_press(self, key):
        if self.on_key_press:
            self.on_key_press(key)
