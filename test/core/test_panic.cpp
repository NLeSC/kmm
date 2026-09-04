#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include "catch2/catch_all.hpp"

#include "kmm/core/panic.hpp"

using namespace kmm;

TEST_CASE("KMM_PANIC") {
    // No way to test for abort using Catch2
    // CHECK(KMM_PANIC("test panic"));
}
