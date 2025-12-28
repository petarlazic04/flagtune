# ===============================
# CONFIG
# ===============================

PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -I..

# C++ build
SRC := main.cpp ../hanning.cpp ../dft.cpp ../mel.cpp ../dct.cpp
BUILD_DIR := build
BIN := $(BUILD_DIR)/main

cpp:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)
	./$(BIN)

BUILD_DIR := build
BIN := $(BUILD_DIR)/main

PYTHON_DIR := .
DATA_DIR := data

# ===============================
# DEFAULT
# ===============================

.PHONY: all
all: config test

# ===============================
# CONFIG (venv + deps)
# ===============================

.PHONY: config
config:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ===============================
# PYTHON
# ===============================

.PHONY: py
py:
	$(PY) test.py

# ===============================
# C++
# ===============================

.PHONY: cpp
cpp:
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)
	./$(BIN)

# ===============================
# TEST (python + cpp)
# ===============================

.PHONY: test
test: py cpp
	@echo "✔ Full pipeline finished"

# ===============================
# CLEAN
# ===============================

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(DATA_DIR)/*.txt

.PHONY: distclean
distclean: clean
	rm -rf $(VENV)
