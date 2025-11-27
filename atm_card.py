import customtkinter as ctk


class ATMCard(ctk.CTkFrame):
    """A draggable ATM card widget that can be inserted into a card slot."""
    
    def __init__(self, master, card_slot=None, on_insert=None, on_remove=None, **kwargs):
        # Set default size and style
        kwargs.setdefault("width", 200)
        kwargs.setdefault("height", 120)
        kwargs.setdefault("fg_color", "#1a5fb4")
        kwargs.setdefault("corner_radius", 10)
        kwargs.setdefault("border_width", 2)
        kwargs.setdefault("border_color", "#0d3b66")

        super().__init__(master, **kwargs)
        
        self.card_slot = card_slot  # Reference to the card slot widget
        self.on_insert = on_insert  # Callback when card is inserted
        self.on_remove = on_remove  # Callback when card is removed
        
        self.is_inserted = False
        self.original_pos = None
        self.home_state = None
        self._drag_data = {"x": 0, "y": 0}
        
        # Prevent the frame from shrinking
        self.pack_propagate(False)
        self.grid_propagate(False)
        
        # Card content frame
        self.card_content = ctk.CTkFrame(self, fg_color="transparent")
        self.card_content.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Bank logo/name
        self.bank_label = ctk.CTkLabel(
            self.card_content,
            text="SECURE BANK",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        self.bank_label.pack(anchor="w")
        
        # Chip (gold rectangle)
        self.chip_frame = ctk.CTkFrame(
            self.card_content,
            width=40,
            height=30,
            fg_color="#FFD700",
            corner_radius=5
        )
        self.chip_frame.pack(anchor="w", pady=5)
        self.chip_frame.pack_propagate(False)
        
        # Card number (masked)
        self.card_number = ctk.CTkLabel(
            self.card_content,
            text="•••• •••• •••• 1234",
            font=("Courier", 12),
            text_color="white"
        )
        self.card_number.pack(anchor="w", pady=(5, 0))
        
        # Cardholder name
        self.holder_name = ctk.CTkLabel(
            self.card_content,
            text="JOHN DOE",
            font=("Arial", 10),
            text_color="white"
        )
        self.holder_name.pack(anchor="w")
        
        # Bind drag events
        self._bind_drag_events(self)
        self._bind_drag_events(self.card_content)
        self._bind_drag_events(self.bank_label)
        self._bind_drag_events(self.chip_frame)
        self._bind_drag_events(self.card_number)
        self._bind_drag_events(self.holder_name)
        
    def _bind_drag_events(self, widget):
        """Bind mouse events for dragging to a widget."""
        widget.bind("<Button-1>", self._on_drag_start)
        widget.bind("<B1-Motion>", self._on_drag_motion)
        widget.bind("<ButtonRelease-1>", self._on_drag_end)
        
    def _on_drag_start(self, event):
        """Record the starting position for drag."""
        if self.is_inserted:
            return  # Don't drag if inserted
            
        # Capture home state on first interaction if not set
        if self.home_state is None:
            self._capture_home_state()
            
        # Record the initial mouse position (absolute) and widget position
        self._drag_data["start_x"] = event.x_root
        self._drag_data["start_y"] = event.y_root
        self._drag_data["widget_x"] = self.winfo_x()
        self._drag_data["widget_y"] = self.winfo_y()
        
        # Lift card above other widgets
        self.lift()

    def _capture_home_state(self):
        """Capture the initial geometry manager state."""
        manager = self.winfo_manager()
        self.home_state = {"manager": manager}
        if manager == "pack":
            self.home_state["info"] = self.pack_info()
        elif manager == "grid":
            self.home_state["info"] = self.grid_info()
        elif manager == "place":
            self.home_state["info"] = self.place_info()
        
        # Also store absolute pos just in case
        self.original_pos = (self.winfo_x(), self.winfo_y())
        
    def _on_drag_motion(self, event):
        """Move the card during drag."""
        if self.is_inserted:
            return
            
        # Calculate delta from start
        dx = event.x_root - self._drag_data["start_x"]
        dy = event.y_root - self._drag_data["start_y"]
        
        # Apply delta to original widget position
        new_x = self._drag_data["widget_x"] + dx
        new_y = self._drag_data["widget_y"] + dy
        
        self.place(x=new_x, y=new_y)
        
    def _on_drag_end(self, event):
        """Check if card was dropped on the card slot."""
        if self.is_inserted:
            return
            
        if self.card_slot:
            # Get card and slot positions
            card_x = self.winfo_rootx()
            card_y = self.winfo_rooty()
            card_width = self.winfo_width()
            card_height = self.winfo_height()
            
            slot_x = self.card_slot.winfo_rootx()
            slot_y = self.card_slot.winfo_rooty()
            slot_width = self.card_slot.winfo_width()
            slot_height = self.card_slot.winfo_height()
            
            # Check for overlap (card center within slot area)
            card_center_x = card_x + card_width // 2
            card_center_y = card_y + card_height // 2
            
            # Make insertion easier by adding a margin around the slot
            margin = 50
            
            if (slot_x - margin <= card_center_x <= slot_x + slot_width + margin and
                slot_y - margin <= card_center_y <= slot_y + slot_height + margin):
                self._insert_card()
            else:
                # Return to original position
                self._return_to_original()
        else:
            self._return_to_original()
            
    def _insert_card(self):
        """Animate card insertion into slot."""
        self.is_inserted = True
        
        # Hide the card completely (simulating full insertion)
        self.place_forget()
            
        if self.on_insert:
            self.on_insert()
            
    def _return_to_original(self):
        """Return card to its original position."""
        if self.home_state:
            manager = self.home_state["manager"]
            info = self.home_state["info"]
            
            # Restore using the original geometry manager
            if manager == "pack":
                # Filter out 'in' if present, as it can cause issues if passed back directly
                if 'in' in info:
                    del info['in']
                self.pack(**info)
            elif manager == "grid":
                if 'in' in info:
                    del info['in']
                self.grid(**info)
            elif manager == "place":
                if 'in' in info:
                    del info['in']
                self.place(**info)
        elif self.original_pos:
            self.place(x=self.original_pos[0], y=self.original_pos[1])
            
    def eject_card(self):
        """Eject the card from the slot."""
        if self.is_inserted:
            self.is_inserted = False
            self.configure(fg_color="#1a5fb4")  # Original color
            self._return_to_original()
            
            if self.on_remove:
                self.on_remove()


class CardSlot(ctk.CTkFrame):
    """A card slot widget where the ATM card can be inserted."""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.configure(
            width=220,
            height=20,
            fg_color="#333333",
            corner_radius=3,
            border_width=2,
            border_color="#555555"
        )
        
        # Slot opening indicator
        self.slot_label = ctk.CTkLabel(
            self,
            text="▼ INSERT CARD ▼",
            font=("Arial", 10),
            text_color="#888888"
        )
        self.slot_label.pack(expand=True)
        
    def set_status(self, inserted: bool):
        """Update slot appearance based on card status."""
        if inserted:
            self.configure(fg_color="#1a5fb4", border_color="#0d3b66")
            self.slot_label.configure(text="CARD INSERTED", text_color="white")
        else:
            self.configure(fg_color="#333333", border_color="#555555")
            self.slot_label.configure(text="▼ INSERT CARD ▼", text_color="#888888")
