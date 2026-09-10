#include "kmm/api/host.hpp"

namespace kmm {

Host Context::host() {
    return Host(m_runtime, m_transaction);
}

}  // namespace kmm
