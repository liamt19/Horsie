
# Compiler and flags
CXX := g++
ARCH := -march=native
_THIS := $(realpath $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
_ROOT := $(_THIS)


LDFLAGS := 


EXE := horsie
EVALFILE := src/incbin/net-013-16l8.bin

GXX_FLAGS:= -mavx -mavx2 -DUSE_PEXT -DUSE_POPCNT -funroll-loops

COMMON_CXXFLAGS := -std=c++23 -DEVALFILE=\"$(EVALFILE)\" $(ARCH) $(GXX_FLAGS)


DEBUG_CXXFLAGS := $(COMMON_CXXFLAGS) -g3 -O0 -DDEBUG -lasan -fsanitize=address,leak,undefined
BUILD_CXXFLAGS := $(COMMON_CXXFLAGS) -O3 -DNDEBUG


# Directories
SRC_DIR := src
BUILD_DIR := build


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
	-rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)
	mkdir $(BUILD_DIR)/nnue
	-rm $(EXE)

# Phony targets
.PHONY: all debug clean pgo-generate

# Disable built-in rules and variables
.SUFFIXES: