# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:11:20 2026

@author: user
"""

import pandas as pd
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from sklearn.feature_selection import mutual_info_classif, RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Core Selector Class (same logic)
# -----------------------------
class HybridFeatureSelector:
    def __init__(self, n_features, method, variance_thresh, corr_thresh, mi_k, random_state=42):
        self.n_features = n_features
        self.method = method
        self.variance_thresh = variance_thresh
        self.corr_thresh = corr_thresh
        self.mi_k = mi_k
        self.random_state = random_state

    def _split(self, df):
        return df.iloc[:, 0], df.iloc[:, 2:], df.iloc[:, 1]

    def _variance_filter(self, X):
        if self.variance_thresh > 0:
            selector = VarianceThreshold(self.variance_thresh)
            X_var = selector.fit_transform(X)
            cols = X.columns[selector.get_support()]
            return pd.DataFrame(X_var, columns=cols)
        return X

    def _correlation_filter(self, X):
        if self.corr_thresh < 1.0:
            corr = X.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            drop_cols = [col for col in upper.columns if any(upper[col] > self.corr_thresh)]
            return X.drop(columns=drop_cols)
        return X

    def _mi(self, X, y, k):
        scores = mutual_info_classif(X, y, random_state=self.random_state)
        idx = np.argsort(scores)[::-1][:k]
        return X.iloc[:, idx]

    def _rf(self, X, y, k):
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        rf.fit(X, y)
        idx = np.argsort(rf.feature_importances_)[::-1][:k]
        return X.iloc[:, idx]

    def _rfe(self, X, y, k):
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        selector = RFE(rf, n_features_to_select=k, step=0.1)
        selector.fit(X, y)
        return X.loc[:, selector.support_]

    def run(self, train_path, test_path, log_func):

        log_func("Loading data...")
        tr = pd.read_csv(train_path)
        ts = pd.read_csv(test_path)

        s_tr, Xtr, ytr = self._split(tr)
        s_ts, Xts, yts = self._split(ts)

        log_func(f"Initial descriptors: {Xtr.shape[1]}")

        # Preprocessing
        Xtr = self._variance_filter(Xtr)
        Xtr = self._correlation_filter(Xtr)

        log_func(f"After filtering: {Xtr.shape[1]}")

        Xts = Xts[Xtr.columns]

        scaler = StandardScaler()
        Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns)
        Xts_s = pd.DataFrame(scaler.transform(Xts), columns=Xts.columns)

        # Selection
        if self.method == "mi":
            Xsel = self._mi(Xtr_s, ytr, self.n_features)

        elif self.method == "rf":
            Xsel = self._rf(Xtr_s, ytr, self.n_features)

        elif self.method == "mi_rf":
            Xtemp = self._mi(Xtr_s, ytr, self.mi_k)
            Xsel = self._rf(Xtemp, ytr, self.n_features)

        elif self.method == "mi_rfe":
            Xtemp = self._mi(Xtr_s, ytr, self.mi_k)
            Xsel = self._rfe(Xtemp, ytr, self.n_features)

        else:
            raise ValueError("Invalid method")

        selected_cols = Xsel.columns

        Xtr_f = Xtr[selected_cols]
        Xts_f = Xts[selected_cols]

        tr_out = pd.concat([s_tr, ytr, Xtr_f], axis=1)
        ts_out = pd.concat([s_ts, yts, Xts_f], axis=1)

        tr_out.to_csv("{}_train_selected.csv".format(self.method), index=False)
        ts_out.to_csv("{}_test_selected.csv".format(self.method), index=False)

        log_func(f"Selected features: {len(selected_cols)}")
        log_func("Saved: train_selected.csv, test_selected.csv")
        log_func("Done!")


# -----------------------------
# Tkinter GUI
# -----------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("QSAR Feature Selector")
        self.root.geometry("600x500")

        self.train_path = tk.StringVar()
        self.test_path = tk.StringVar()

        self.n_features = tk.IntVar(value=15)
        self.method = tk.StringVar(value="mi_rf")
        self.variance = tk.DoubleVar(value=0.001)
        self.correlation = tk.DoubleVar(value=0.95)
        self.mi_k = tk.IntVar(value=200)

        self._build_ui()

    def _build_ui(self):

        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        # File selectors
        ttk.Label(frame, text="Training CSV").grid(row=0, column=0)
        ttk.Entry(frame, textvariable=self.train_path, width=40).grid(row=0, column=1)
        ttk.Button(frame, text="Browse", command=self.browse_train).grid(row=0, column=2)

        ttk.Label(frame, text="Test CSV").grid(row=1, column=0)
        ttk.Entry(frame, textvariable=self.test_path, width=40).grid(row=1, column=1)
        ttk.Button(frame, text="Browse", command=self.browse_test).grid(row=1, column=2)

        # Parameters
        ttk.Label(frame, text="No. of Features").grid(row=2, column=0)
        ttk.Entry(frame, textvariable=self.n_features).grid(row=2, column=1)

        ttk.Label(frame, text="Method").grid(row=3, column=0)
        ttk.Combobox(frame, textvariable=self.method,
                     values=["mi", "rf", "mi_rf", "mi_rfe"]).grid(row=3, column=1)

        ttk.Label(frame, text="Variance Threshold").grid(row=4, column=0)
        ttk.Entry(frame, textvariable=self.variance).grid(row=4, column=1)

        ttk.Label(frame, text="Correlation Cutoff").grid(row=5, column=0)
        ttk.Entry(frame, textvariable=self.correlation).grid(row=5, column=1)

        ttk.Label(frame, text="MI prefilter (k)").grid(row=6, column=0)
        ttk.Entry(frame, textvariable=self.mi_k).grid(row=6, column=1)

        # Run button
        ttk.Button(frame, text="Run", command=self.run_thread).grid(row=7, column=1, pady=10)

        # Log box
        self.log_box = tk.Text(frame, height=15)
        self.log_box.grid(row=8, column=0, columnspan=3, sticky="nsew")

    def browse_train(self):
        self.train_path.set(filedialog.askopenfilename(filetypes=[("CSV", "*.csv")]))
        print(self.train_path)

    def browse_test(self):
        self.test_path.set(filedialog.askopenfilename(filetypes=[("CSV", "*.csv")]))

    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def run_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()

    def run(self):
        try:
            fs = HybridFeatureSelector(
                self.n_features.get(),
                self.method.get(),
                self.variance.get(),
                self.correlation.get(),
                self.mi_k.get()
            )

            fs.run(self.train_path.get(), self.test_path.get(), self.log)

        except Exception as e:
            messagebox.showerror("Error", str(e))


# -----------------------------
# Launch
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()