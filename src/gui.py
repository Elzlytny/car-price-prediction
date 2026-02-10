import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib

# ================= Load Model =================
model = joblib.load('models/rf_model.pkl')
columns = joblib.load('models/model_columns.pkl')

# ================= Window =================
root = tk.Tk()
root.title("Car Price Prediction")
root.geometry("520x600")
root.minsize(450, 550)
root.configure(bg="#080627")

# # Responsive config
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# ================= Main Frame =================
main_frame = tk.Frame(root, bg="#050222")
main_frame.grid(row=0, column=0, sticky="nsew", padx=25, pady=25) 
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(8, weight=1)

# ================= Title =================
title = tk.Label(
    main_frame,
    text="🚗 Car Price Prediction",
    font=("Segoe UI", 20, "bold"),
    bg="#210249",
    fg="#ffffff"
)
title.grid(row=0, column=0, columnspan=2, pady=(10, 30))

# ================= Styles =================
style = ttk.Style()
style.theme_use("default")

style.configure(
    "TCombobox",
    fieldbackground="#1e1e1e",
    background="#1e1e1e",
    foreground="black",
)

def create_input(label, row):
    tk.Label(
        main_frame,
        text=label,
        font=("Segoe UI", 11),
        bg="#210249",
        fg="#bbbbbb"
    ).grid(row=row, column=0, sticky="news", pady=12, padx=15)

    entry = tk.Entry(
        main_frame,
        font=("Segoe UI", 11),
        bg="#1e1e1e",
        fg="white",
        insertbackground="white",
        relief="flat"
    )
    entry.grid(row=row, column=1, sticky="nsew", pady=12, padx=15)
    return entry

# ================= Inputs =================
year_entry = create_input("Year", 1)
km_entry = create_input("Kilometer", 2)
engine_entry = create_input("Engine (cc)", 3)
power_entry = create_input("Max Power (bhp)", 4)

# ================= Owner Dropdown =================
tk.Label(
        main_frame,
        text="Owner",
        font=("Segoe UI", 11),
        bg="#210249",
        fg="#bbbbbb"
    ).grid(row=5, column=0, sticky="news", pady=12, padx=15)

owner_combo = ttk.Combobox(
    main_frame,
    values=["First", "Second", "Third", "Fourth & Above"],
    state="readonly",
    font=("Segoe UI", 11)
)
owner_combo.current(0)
owner_combo.grid(row=5, column=1, sticky="nsew", pady=12, padx=15)

owner_map = {
    "First": 1,
    "Second": 2,
    "Third": 3,
    "Fourth & Above": 4
}

# ================= Result =================
result_label = tk.Label(
    main_frame,
    text="Predicted Price: ---",
    font=("Segoe UI", 15, "bold"),
    bg="#210249",
    fg="#00e676"
)
result_label.grid(row=6, column=0, columnspan=2, pady=30)

# ================= Prediction Function =================
def predict_price():
    try:
        data = {
            'Year': int(year_entry.get()),
            'Kilometer': int(km_entry.get()),
            'Engine': float(engine_entry.get()),
            'Max Power': float(power_entry.get()),
            'Owner': owner_map[owner_combo.get()]
        }

        df = pd.DataFrame([data])
        df = df.reindex(columns=columns, fill_value=0)

        price = model.predict(df)[0]
        result_label.config(text=f"Predicted Price: {int(price):,} LE")

    except:
        result_label.config(text="❌ Please enter valid data", fg="#ff5252")

# ================= Button =================
def on_enter(e):
    predict_btn.config(bg="#210249")

def on_leave(e):
    predict_btn.config(bg="#220c57")

predict_btn = tk.Button(
    main_frame,
    text="Predict Price",
    font=("Segoe UI", 13, "bold"),
    bg="#1a1050",
    fg="white",
    relief="flat",
    height=2,
    command=predict_price
)
predict_btn.grid(row=7, column=0, columnspan=2, pady=10)

predict_btn.bind("<Enter>", on_enter)
predict_btn.bind("<Leave>", on_leave)

# ================= Footer =================
footer = tk.Label(
    main_frame,
    text="Machine Learning Regression Project",
    font=("Segoe UI", 9),
    bg="#210249",
    fg="#666666"
)
footer.grid(row=8, column=0, columnspan=2, pady=(20, 5))

root.mainloop()
