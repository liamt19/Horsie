
# Compiler and flags
CXX := g++
PGO = off

EXE := horsie

SOURCES := src/bitboard.cpp src/cuckoo.cpp src/Horsie.cpp src/movegen.cpp src/position.cpp src/precomputed.cpp src/search.cpp src/threadpool.cpp src/tt.cpp src/zobrist.cpp src/nnue/nn.cpp src/3rdparty/zstd/zstddeclib.c

COMPILER_VERSION := $(shell $(CXX) --version)
ARCH_DEFINES := $(shell echo | $(CXX) -march=native -E -dM -)


ifeq ($(UNAME_S),Darwin)
	DEFAULT_NET := $(shell cat network.txt)
else
	DEFAULT_NET := $(file < network.txt)
endif

ifndef EVALFILE
    EVALFILE = $(DEFAULT_NET).bin
    NO_EVALFILE_SET = true
endif


CXXFLAGS:= -std=c++23 -g -O3 -DNDEBUG -DEVALFILE=\"$(EVALFILE)\" -DUSE_PEXT -DUSE_POPCNT -funroll-loops
DEBUG_CXXFLAGS := $(COMMON_CXXFLAGS) -g3 -O0 -DDEBUG -lasan -fsanitize=address,leak,undefined

CXXFLAGS_NATIVE := -march=native
CXXFLAGS_AVX2_BMI2 := -march=haswell -mtune=haswell

ifneq ($(findstring __AVX512BW__, $(ARCH_DEFINES)),)
CXXFLAGS_NATIVE += -DAVX512
endif
ifneq ($(findstring __AVX2__, $(ARCH_DEFINES)),)
CXXFLAGS_NATIVE += -DAVX256
endif
ifneq ($(findstring __AVX__, $(ARCH_DEFINES)),)
CXXFLAGS_NATIVE += -DAVX128
endif


ifeq ($(CXX),clang++)
	STACK_SIZE := -Wl,/STACK:12582912
else
	STACK_SIZE := -Wl,--stack,12194304
endif

ifeq ($(OS),Windows_NT) 
	CXXFLAGS += -fuse-ld=lld -static
	RM_FILE_CMD = del
	LDFLAGS += $(STACK_SIZE)
	SUFFIX := .exe
else
	RM_FILE_CMD = rm
endif

#	https://github.com/Ciekce/Stormphrax/blob/main/Makefile

ifneq (, $(findstring clang,$(COMPILER_VERSION)))
PGO_GENERATE := -fprofile-instr-generate
PGO_MERGE := llvm-profdata merge -output=horsie_p.profdata *.profraw
PGO_USE := -fprofile-instr-use=horsie_p.profdata
PGO_CLEAN := $(RM_FILE_CMD) *.profraw horsie_p.profdata
else
PGO_GENERATE := -fprofile-generate
PGO_MERGE := 
PGO_USE := -fprofile-use
PGO_CLEAN := $(RM_FILE_CMD) *.gcda
endif

PROFILE_OUT = horsie$(SUFFIX)


ifneq ($(PGO),on)
define build
    $(CXX) $(CXXFLAGS) $(CXXFLAGS_$1) $(LDFLAGS) -o $(EXE)$(if $(NO_EXE_SET),-$2)$(SUFFIX) $(filter-out $(EVALFILE),$^)
endef
else
define build
    $(CXX) $(CXXFLAGS) $(CXXFLAGS_$1) $(LDFLAGS) -o $(PROFILE_OUT) $(PGO_GENERATE) $(filter-out $(EVALFILE),$^)
    ./$(PROFILE_OUT) bench 13
    $(RM_FILE_CMD) $(PROFILE_OUT)
	$(PGO_MERGE)
    $(CXX) $(CXXFLAGS) $(CXXFLAGS_$1) $(LDFLAGS) -o $(EXE)$(if $(NO_EXE_SET),-$2)$(SUFFIX) $(PGO_USE) $(filter-out $(EVALFILE),$^)
	$(PGO_CLEAN)
endef
endif


release: avx2-bmi2
all: native release

.PHONY: all

.DEFAULT_GOAL := native

ifdef NO_EVALFILE_SET
$(EVALFILE):
	$(info Downloading default network $(DEFAULT_NET).bin)
	curl -sOL https://github.com/liamt19/lizard-nets/releases/download/$(DEFAULT_NET)/$(DEFAULT_NET).bin

download-net: $(EVALFILE)
endif

$(EXE): $(EVALFILE) $(SOURCES)
	$(call build,NATIVE,native)

native: $(EXE)

avx2-bmi2: $(EVALFILE) $(SOURCES)
	$(call build,AVX2_BMI2,avx2-bmi2)

clean: