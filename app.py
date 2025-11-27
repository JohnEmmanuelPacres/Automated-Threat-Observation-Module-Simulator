import threading
import queue
import time
import cv2

import customtkinter as ctk
from PIL import Image, ImageTk

from mmwave_simul import MMSimulator
from atm_keypad import ATMKeypad
from atm_card import ATMCard, CardSlot
import numpy as np
import traceback

class App(ctk.CTk):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.title("mmWave Simulator UI")
		self.geometry("1360x1000")

		ctk.set_appearance_mode("dark")
		ctk.set_default_color_theme("blue")

		self.grid_columnconfigure(0, weight=1)
		self.grid_columnconfigure(1, weight=0)
		self.grid_rowconfigure(1, weight=1)
		
		self.sim = None

		# UI elements
		# Control Panel (Top)
		self.controls_frame = ctk.CTkFrame(self)
		self.controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
		self.controls_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

		self.start_btn = ctk.CTkButton(self.controls_frame, text="CAMERA ON", command=self.start_sim, state="normal", fg_color="green")
		self.start_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

		self.stop_btn = ctk.CTkButton(self.controls_frame, text="CAMERA OFF", command=self.stop_sim, state="disabled", fg_color="gray")
		self.stop_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

		self.weapon_btn = ctk.CTkButton(self.controls_frame, text="Weapon: OFF", state="disabled", command=self.toggle_weapon)
		self.weapon_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

		self.coercion_btn = ctk.CTkButton(self.controls_frame, text="Simulate Coercion", state="disabled", command=self.simulate_coercion, fg_color="blue")
		self.coercion_btn.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

		self.view_mode_var = ctk.StringVar(value="Full Debug")
		self.view_mode_menu = ctk.CTkOptionMenu(self.controls_frame, values=["Full Debug", "Standard Monitoring", "Privacy Mode", "Security Mode"],
												command=self.change_view_mode, variable=self.view_mode_var, state="disabled")
		self.view_mode_menu.grid(row=0, column=4, padx=5, pady=5, sticky="ew")

		# Video Display (Middle)
		self.video_label = ctk.CTkLabel(self, text="Press CAMERA ON to run simulation", width=960, height=720)
		self.video_label.grid(row=1, column=0, padx=10, pady=10)

		# Status Bar (Bottom)
		self.status_label = ctk.CTkLabel(self, text="Status: CAMERA OFF")
		self.status_label.grid(row=2, column=0, padx=10, pady=(0,10), sticky="w")

		# Safety Alert UI (Hidden initially)
		self.is_safety_mode = False
		self.last_safety_confirm_time = 0
		self.safety_cooldown = 5.0  # 5 seconds buffer
		self.safety_frame = ctk.CTkFrame(self, fg_color="#8B0000", corner_radius=20, border_width=2, border_color="white")
		
		self.safety_label = ctk.CTkLabel(self.safety_frame, text="POTENTIAL THREAT DETECTED\nARE YOU SAFE?", 
										 font=("Arial", 32, "bold"), text_color="white")
		# self.safety_label.pack(pady=(30, 20), padx=40) # Packed dynamically

		self.safety_label2 = ctk.CTkLabel(self.safety_frame, text="SOMEONE IS WATCHING YOU?\n CONTINUE TRANSACTION?",
										 font=("Arial", 32, "bold"), text_color="white")
		# self.safety_label2.pack(pady=(30, 20), padx=40) # Packed dynamically

		self.btn_frame = ctk.CTkFrame(self.safety_frame, fg_color="transparent")
		self.btn_frame.pack(pady=20)

		self.safe_btn = ctk.CTkButton(self.btn_frame, text="YES, I AM SAFE", command=self.confirm_safe, 
									  fg_color="green", hover_color="darkgreen", width=180, height=60, font=("Arial", 16, "bold"))
		self.safe_btn.pack(side="left", padx=20)

		self.emergency_btn = ctk.CTkButton(self.btn_frame, text="EMERGENCY ALERT", command=self.trigger_emergency, 
										   fg_color="red", hover_color="darkred", width=180, height=60, font=("Arial", 16, "bold"))
		self.emergency_btn.pack(side="right", padx=20)

		self._imgtk = None
		self._update_job = None
	
		self.protocol("WM_DELETE_WINDOW", self._on_close)

		self.setup_atm_ui()
		self.show_loading_screen()

	def show_loading_screen(self):
		self.loading_frame = ctk.CTkFrame(self, fg_color="#101010")
		self.loading_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

		self.loading_label = ctk.CTkLabel(self.loading_frame, text="ATOM SIMULATOR\nInitializing System...", font=("Arial", 24, "bold"))
		self.loading_label.place(relx=0.5, rely=0.4, anchor="center")

		self.progress_bar = ctk.CTkProgressBar(self.loading_frame, width=400, mode="determinate")
		self.progress_bar.place(relx=0.5, rely=0.5, anchor="center")
		self.progress_bar.set(0)

		self.loading_status = ctk.CTkLabel(self.loading_frame, text="Starting...", font=("Arial", 12))
		self.loading_status.place(relx=0.5, rely=0.55, anchor="center")

		self.loading_step = 0
		self.after(500, self.update_loading)

	def update_loading(self):
		steps = [
			(0.1, "Loading UI Components..."),
			(0.3, "Initializing Camera Interface..."),
			(0.5, "Loading AI Models (YOLOv8)..."),
			(0.7, "Configuring mmWave Simulation..."),
			(0.9, "Establishing Secure Connection..."),
			(1.0, "System Ready")
		]

		if self.loading_step < len(steps):
			progress, text = steps[self.loading_step]
			self.progress_bar.set(progress)
			self.loading_status.configure(text=text)
			self.loading_step += 1
			self.after(600, self.update_loading)
		else:
			self.loading_frame.destroy()

	def setup_atm_ui(self):
		self.atm_frame = ctk.CTkFrame(self, width=300)
		self.atm_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
		
		# Card Slot
		self.card_slot = CardSlot(self.atm_frame)
		self.card_slot.pack(padx=10, pady=(20, 5))
		
		# ATM Screen
		self.atm_screen = ctk.CTkLabel(
			self.atm_frame, 
			text="WELCOME\nInsert Card to Begin",
			font=("Courier", 20, "bold"),
			width=280,
			height=150,
			fg_color="black",
			text_color="#00FF00",
			corner_radius=5
		)
		self.atm_screen.pack(padx=10, pady=10)
		
		# Eject Card Button
		self.eject_btn = ctk.CTkButton(
			self.atm_frame,
			text="EJECT CARD",
			command=self.eject_card,
			fg_color="#555555",
			hover_color="#666666",
			state="disabled",
			width=200,
			height=30
		)
		self.eject_btn.pack(padx=10, pady=5)
		
		# Keypad
		self.keypad = ATMKeypad(self.atm_frame, on_key_press=self.on_atm_key)
		self.keypad.pack(padx=10, pady=10)
		
		# ATM Card (draggable) - placed at bottom of ATM frame
		self.card_holder_label = ctk.CTkLabel(
			self.atm_frame,
			text="↓ Drag card to slot above ↓",
			font=("Arial", 10),
			text_color="#888888"
		)
		self.card_holder_label.pack(pady=(20, 5))
		
		self.atm_card = ATMCard(
			self.atm_frame,
			card_slot=self.card_slot,
			on_insert=self.on_card_insert,
			on_remove=self.on_card_remove
		)
		self.atm_card.pack(padx=10, pady=10)
		
		# Transaction State
		self.atm_state = "WELCOME" # WELCOME, PIN, AMOUNT, PROCESSING, RESULT
		self.input_buffer = ""
		self.card_inserted = False
	
	def on_card_insert(self):
		"""Called when ATM card is inserted into slot."""
		self.card_inserted = True
		if self.sim:
			self.sim.set_card_inserted(True)
		self.card_slot.set_status(True)
		self.eject_btn.configure(state="normal", fg_color="#1F6AA5")
		self.atm_state = "PIN"
		self.input_buffer = ""
		self.update_atm_screen("CARD ACCEPTED\nEnter PIN:")
		
	def on_card_remove(self):
		"""Called when ATM card is removed from slot."""
		self.card_inserted = False
		if self.sim:
			self.sim.set_card_inserted(False)
		self.card_slot.set_status(False)
		self.eject_btn.configure(state="disabled", fg_color="#555555")
		self.atm_state = "WELCOME"
		self.input_buffer = ""
		self.update_atm_screen("WELCOME\nInsert Card to Begin")
		
	def eject_card(self):
		"""Eject the ATM card."""
		if self.atm_card:
			self.atm_card.eject_card()
	
	def on_atm_key(self, key):
		# Check if card is inserted first
		if not self.card_inserted and self.atm_state == "WELCOME":
			self.update_atm_screen("Please Insert Card\nFirst")
			self.after(1500, lambda: self.update_atm_screen("WELCOME\nInsert Card to Begin"))
			return
		
		if key == "CANCEL":
			self.atm_state = "PIN" if self.card_inserted else "WELCOME"
			self.input_buffer = ""
			if self.card_inserted:
				self.update_atm_screen("CARD ACCEPTED\nEnter PIN:")
			else:
				self.update_atm_screen("WELCOME\nInsert Card to Begin")
			return
			
		if key == "CLEAR":
			self.input_buffer = ""
			self.update_atm_screen_display()
			return
			
		if key == "ENTER":
			self.process_enter()
			return
			
		# Digits
		if self.atm_state in ["PIN", "AMOUNT"]:
			if len(self.input_buffer) < 10:
				self.input_buffer += key
				self.update_atm_screen_display()
				
	def update_atm_screen_display(self):
		if self.atm_state == "PIN":
			display_text = "ENTER PIN:\n" + "*" * len(self.input_buffer)
		elif self.atm_state == "AMOUNT":
			display_text = "ENTER AMOUNT:\n$" + self.input_buffer
		else:
			display_text = self.input_buffer
		self.update_atm_screen(display_text)
		
	def update_atm_screen(self, text):
		self.atm_screen.configure(text=text)
		
	def process_enter(self):
		if self.atm_state == "PIN":
			if len(self.input_buffer) == 4: # Simple validation
				self.atm_state = "AMOUNT"
				self.input_buffer = ""
				self.update_atm_screen("ENTER AMOUNT:\n$")
			else:
				self.update_atm_screen("INVALID PIN\nTry Again")
				self.input_buffer = ""
				self.after(2000, lambda: self.update_atm_screen_display())
				
		elif self.atm_state == "AMOUNT":
			if self.input_buffer:
				amount = self.input_buffer
				self.atm_state = "PROCESSING"
				self.update_atm_screen("PROCESSING...")
				self.after(2000, lambda: self.finish_transaction(amount))
				
	def finish_transaction(self, amount):
		self.update_atm_screen(f"WITHDRAWAL SUCCESS\n${amount}\n\nPlease take cash\nand card")
		self.after(4000, self.reset_atm)
		
	def reset_atm(self):
		# Eject card after transaction
		if self.card_inserted:
			self.eject_card()
		self.atm_state = "WELCOME"
		self.input_buffer = ""
		self.update_atm_screen("WELCOME\nInsert Card to Begin")

	def start_sim(self):
		if self.sim and self.sim._thread and self.sim._thread.is_alive():
			return
		# create simulator and start
		self.sim = MMSimulator()
		self.sim.set_view_mode(self.view_mode_var.get())
		self.sim.set_card_inserted(self.card_inserted)
		self.sim.start()
		self.status_label.configure(text="Status: CAMERA ON")
		self.start_btn.configure(state="disabled")
		self.stop_btn.configure(state="normal", fg_color="red")
		self.weapon_btn.configure(state="normal")
		self.coercion_btn.configure(state="normal", text="Simulate Coercion", fg_color="orange")
		self.view_mode_menu.configure(state="normal")
		self._schedule_update()


	def stop_sim(self):
		if self.sim:
			self.sim.stop()
			self.sim = None
		self.status_label.configure(text="Status: CAMERA OFF")
		self.start_btn.configure(state="normal")
		self.stop_btn.configure(state="disabled", fg_color="gray")
		self.weapon_btn.configure(state="disabled")
		self.coercion_btn.configure(state="disabled", text="Simulate Coercion", fg_color="orange")
		self.view_mode_menu.configure(state="disabled")
		if self._update_job:
			self.after_cancel(self._update_job)
			self._update_job = None

	def toggle_weapon(self):
		if not self.sim:
			return
		self.sim.toggle_weapon_mode()
		if self.sim.weapon_mode:
			self.weapon_btn.configure(text="Weapon: ON", fg_color="red")
		else:
			self.weapon_btn.configure(text="Weapon: OFF", fg_color="#1F6AA5")

	def simulate_coercion(self):
		if self.sim:
			self.sim.trigger_coercion()
			self.coercion_btn.configure(text="Coercion Active", fg_color="red")

	def change_view_mode(self, choice):
		if self.sim:
			self.sim.set_view_mode(choice)

	def _schedule_update(self):
		self._update_frame()

	def _update_frame(self):
		if self.sim:
			try:
				data = self.sim.frame_queue.get_nowait()
				if isinstance(data, tuple):
					frame, threat_detected = data
				else:
					frame = data
					threat_detected = False
			except queue.Empty:
				frame = None
				threat_detected = False

			if frame is not None:
				# frame is expected to be a BGR numpy array; validate and convert safely
				try:
					if frame is None:
						raise ValueError("Empty frame")
					if not isinstance(frame, np.ndarray):
						raise ValueError(f"Frame is not a numpy array: {type(frame)}")
					
					# Check threat
					if threat_detected:
						if threat_detected == "WEAPON_LOCK":
							if not self.is_safety_mode or self.atm_state != "LOCKED":
								self.lock_atm_and_alert()
						elif threat_detected == "HIGH_HR":
							if not self.is_safety_mode:
								self.is_safety_mode = True
								self.show_safety_alert("HIGH_HR")
						elif not self.is_safety_mode:
							# Dynamic buffer: 30s for peeking, 0s (immediate) for weapons/hands
							required_buffer = 30.0 if threat_detected == "PEEKING" else 0.0

							if time.time() - self.last_safety_confirm_time > required_buffer:
								self.is_safety_mode = True
								self.show_safety_alert(threat_detected)

					if frame.ndim == 2:
						# single channel: convert to RGB by stacking
						img_arr = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
					elif frame.ndim == 3 and frame.shape[2] == 3:
						img_arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					elif frame.ndim == 3 and frame.shape[2] == 4:
						img_arr = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
					else:
						raise ValueError(f"Unsupported frame shape: {frame.shape}")

					# Blur if in safety mode
					if self.is_safety_mode:
						img_arr = cv2.GaussianBlur(img_arr, (99, 99), 30)

					pil = Image.fromarray(img_arr)
					pil = pil.resize((960, 720), Image.Resampling.LANCZOS)
					self._imgtk = ImageTk.PhotoImage(pil)
					self.video_label.configure(image=self._imgtk, text="")
				except Exception as e:
					# show error in status label and print traceback for debugging
					self.status_label.configure(text=f"Frame error: {e}")
					traceback.print_exc()

		# schedule next update
		self._update_job = self.after(30, self._update_frame)

	def lock_atm_and_alert(self):
		self.is_safety_mode = True
		self.atm_state = "LOCKED"
		self.update_atm_screen("SYSTEM LOCKED\nTHREAT DETECTED")
		
		# Show safety alert immediately
		self.show_safety_alert("WEAPON")
		
		# Trigger emergency (sound + visual)
		self.trigger_emergency()
		
		# Eject card if inserted
		if self.card_inserted:
			self.eject_card()

	def show_safety_alert(self, threat_type="THREAT"):
		# Show appropriate label
		if threat_type == "PEEKING":
			self.safety_label2.pack(pady=(30, 20), padx=40)
		elif threat_type == "HIGH_HR":
			self.safety_label.configure(text="HIGH STRESS DETECTED\nARE YOU UNDER DURESS?")
			self.safety_label.pack(pady=(30, 20), padx=40)
		else:
			self.safety_label.configure(text="POTENTIAL THREAT DETECTED\nARE YOU SAFE?")
			self.safety_label.pack(pady=(30, 20), padx=40)

		# Place the safety frame over the video
		self.safety_frame.place(relx=0.5, rely=0.5, anchor="center")
		# Bring to front
		self.safety_frame.lift()

	def confirm_safe(self):
		if self.sim:
			self.sim.clear_recording()
			# Reset coercion mode if it was active
			if self.sim.vitals_sim.coercion_active:
				self.sim.vitals_sim.set_coercion(False)
				self.coercion_btn.configure(text="Simulate Coercion", fg_color="orange")

		self.is_safety_mode = False
		self.safety_frame.place_forget()
		self.safety_label.pack_forget()
		self.safety_label2.pack_forget()
		# Optional: Add a temporary ignore period for threats to prevent immediate re-trigger
		self.last_safety_confirm_time = time.time()

	def trigger_emergency(self):
		# Recording continues until alarm stops
		# Play loud sound (simulated)
		print("EMERGENCY TRIGGERED!")
		
		# Visual feedback
		self.safety_label2.pack_forget() # Hide the "Watching you" label if present
		self.safety_label.pack(pady=(30, 20), padx=40) # Ensure main label is visible

		self.safety_label.configure(text="AUTHORITIES CONTACTED\nSTAY CALM")
		self.safe_btn.pack_forget()
		self.emergency_btn.configure(text="ALARM ACTIVE", state="disabled")
		
		# Play sound in a thread
		threading.Thread(target=self._play_alarm_sound, daemon=True).start()

	def _play_alarm_sound(self):
		try:
			# Windows beep or powershell sound
			for _ in range(5):
				import winsound
				winsound.Beep(1000, 500)
				time.sleep(0.1)
				winsound.Beep(1500, 500)
				time.sleep(0.1)
		except:
			pass
		
		# Reset UI after some time or keep it? 
		# For demo, let's reset after 5 seconds
		time.sleep(3)
		self.after(0, self._reset_safety_ui)

	def _reset_safety_ui(self):
		if self.sim:
			self.sim.save_recording()
			# Reset coercion mode if it was active
			if self.sim.vitals_sim.coercion_active:
				self.sim.vitals_sim.set_coercion(False)
				self.coercion_btn.configure(text="Simulate Coercion", fg_color="orange")

		self.is_safety_mode = False
		self.safety_frame.place_forget()
		self.safety_label.pack_forget()
		self.safety_label2.pack_forget()
		self.safety_label.configure(text="POTENTIAL THREAT DETECTED\nARE YOU SAFE?")
		self.safe_btn.pack(side="left", padx=20)
		self.emergency_btn.configure(text="EMERGENCY ALERT", state="normal")

	def _on_close(self):
		self.stop_sim()
		self.destroy()


if __name__ == '__main__':
	app = App()
	app.mainloop()



