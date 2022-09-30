# AMReX
DIM = 2
COMP = gnu
PRECISION = DOUBLE

# Profiling
PROFILE = FALSE
TINY_PROFILE = FALSE
COMM_PROFILE = FALSE
TRACE_PROFILE = FALSE
MEM_PROFILE = FALSE
USE_GPROF = FALSE

# Performance
USE_MPI = TRUE
USE_OMP = FALSE
USE_CUDA = FALSE
USE_HIP = FALSE
USE_DPCPP = FALSE

# Debugging
DEBUG = FALSE
FSANITIZER = FALSE
THREAD_SANITIZER = FALSE

# PeleC
USE_REACT = TRUE
USE_EB = FALSE
Eos_Model := Fuego
Transport_Model := Simple
Chemistry_Model := grimech12

# GNU Make
Bpack := ./Make.package
Blocs := .
PELEC_HOME := /home/suryanarayan/Pele/Pele_dev/PeleC
include $(PELEC_HOME)/Exec/Make.PeleC
