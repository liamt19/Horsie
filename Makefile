
# Compiler and flags
CXX := g++
ARCH := -march=native
_THIS := $(realpath $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
_ROOT := $(_THIS)


LDFLAGS := 


EXE := horsie

ifeq ($(UNAME_S),Darwin)
	DEFAULT_NET := $(shell cat network.txt)
else
	DEFAULT_NET := $(file < network.txt)
endif

ifndef EVALFILE
    EVALFILE = $(DEFAULT_NET).bin
    NO_EVALFILE_SET = true
endif


GXX_FLAGS:= -mavx -mavx2 -DUSE_PEXT -DUSE_POPCNT -funroll-loops

COMMON_CXXFLAGS := -std=c++23 -DEVALFILE=\"$(EVALFILE)\" $(ARCH) $(GXX_FLAGS)


DEBUG_CXXFLAGS := $(COMMON_CXXFLAGS) -g3 -O0 -DDEBUG -lasan -fsanitize=address,leak,undefined
BUILD_CXXFLAGS := $(COMMON_CXXFLAGS) -O3 -DNDEBUG


# Directories
SRC_DIR := src
BUILD_DIR := build


ifeq ($(OS),Windows_NT) 
	NNUE_DIR_CMD = cd $(BUILD_DIR) && mkdir nnue && cd ..
	RM_FILE_CMD = del
	RM_FOLDER_CMD = rmdir /s /q
	LDFLAGS += -Wl,--stack,12194304
else
	NNUE_DIR_CMD = -mkdir $(BUILD_DIR)/nnue
	RM_FILE_CMD = rm
	RM_FOLDER_CMD = rm -rf
endif


# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
SRCS += $(wildcard $(SRC_DIR)/nnue/*.cpp)

OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))


# Append .exe to the binary name on Windows
ifeq ($(OS),Windows_NT)
	CXXFLAGS += -fuse-ld=lld
    override EXE := $(EXE).exe
endif


# Default target
all: CXXFLAGS += $(BUILD_CXXFLAGS)
all: clean
all: $(EXE)

ifdef NO_EVALFILE_SET
$(EVALFILE):
	$(info Downloading default network $(DEFAULT_NET).bin)
	curl -sOL https://github.com/liamt19/lizard-nets/releases/download/$(DEFAULT_NET)/$(DEFAULT_NET).bin

download-net: $(EVALFILE)
endif

# Rule to build the target binary
$(EXE): $(EVALFILE) $(OBJS)
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
	-$(RM_FOLDER_CMD) $(BUILD_DIR)
	-mkdir $(BUILD_DIR)
	$(NNUE_DIR_CMD)
	-$(RM_FILE_CMD) $(EXE)

# Phony targets
.PHONY: all debug clean pgo-generate

# Disable built-in rules and variables
.SUFFIXES: