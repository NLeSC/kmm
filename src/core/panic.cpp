#include <cstdio>
#include <cstdlib>

// For POSIX stack trace (Linux, macOS, etc.)
// Remove if not available on your platform
#ifdef __GNUC__
    #include <execinfo.h>
#endif

namespace kmm {

[[noreturn]] void panic(const char* file, int line, const char* message) {
    fprintf(stderr, "\nPANIC TRIGGERED\n");
    fprintf(stderr, "  location: %s:%d\n", file, line);
    fprintf(stderr, "  message: %s\n", message);
    fprintf(stderr, "  stack trace:\n");

#ifdef __GNUC__
    // Attempt to capture and print a backtrace
    void* callstack[256];
    int nframes = backtrace(callstack, 256);
    char** symbols = backtrace_symbols(callstack, nframes);

    if (symbols != nullptr) {
        for (int i = 0; i < nframes; i++) {
            fprintf(stderr, "    %s\n", symbols[i]);
        }
    } else {
        fprintf(stderr, "    ??? <backtrace_symbols() failed>\n");
    }
#else
    fprintf(stderr, "    ??? <backtrace_symbols() not supported>\n");
#endif

    fflush(stderr);

    // Use abort() to generate a core dump
    abort();
}

}  // namespace kmm
