# ===============================
# CONFIG
# ===============================
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

REQUIREMENTS = requirements.txt

# ===============================
# TARGETS
# ===============================

# 1️⃣ Config → venv + instalacija paketa
.PHONY: config
config:
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

# 2️⃣ Generisanje MFCC matrica
.PHONY: mfcc
mfcc: config
	$(PYTHON) generate_mfcc.py

# 3️⃣ Trening NN modela
.PHONY: train
train: config
	$(PYTHON) train.py

# 4️⃣ Real-time simulacija / inference
.PHONY: infer
infer: config
	$(PYTHON) infer.py

# 5️⃣ Default → sve pokrene
.PHONY: all
all: mfcc train infer

# 6️⃣ Čišćenje venv i generisanih fajlova (opciono)
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)
	rm -rf data/mfcc/*.npy
	rm -rf model/*.h5 model/*.tflite
