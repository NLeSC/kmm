
#ifdef KMM_USE_CUDA
    #include "../backends/cuda.cpp"
#elif KMM_USE_HIP
    #include "../backends/hip.cpp"
#else
    #include "../backends/cpu.cpp"
#endif