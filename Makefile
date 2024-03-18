
# Compiler and flags
CXX := g++
ARCH := -march=native
_THIS     := $(realpath $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
_ROOT     := $(_THIS)

LDFLAGS := 

# Debug compiler flags
DEBUG_CXXFLAGS := -std=c++20 -g3 -O1 -DDEBUG -fsanitize=address -fsanitize=undefined 

#BUILD_CXXFLAGS := -DNDEBUG -O3

BUILD_CXXFLAGS := -std=c++20 -g3 -O1 -DDEBUG

# Directories
SRC_DIR := src
BUILD_DIR := build

# Source files
SRCS := $(filter-out , $(wildcard src/*.cpp))
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))


EXE := horsie

# Append .exe to the binary name on Windows
ifeq ($(OS),Windows_NT)
	CXXFLAGS += -fuse-ld=lld
    override EXE := $(EXE).exe
endif

# Default target
all: CXXFLAGS += $(BUILD_CXXFLAGS)
all: clean
all: $(EXE) 

# Rule to build the target binary
$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJS)

# Rule to build object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Create directories if they don't exist
$(BUILD_DIR):
	-mkdir $@

# Debug target
debug: CXXFLAGS += $(DEBUG_CXXFLAGS)
debug: $(EXE)

# Clean the build
clean:
	-rmdir /s /q $(BUILD_DIR)
	-del $(EXE)

# Phony targets
.PHONY: all debug clean pgo-generate

# Disable built-in rules and variables
.SUFFIXES: