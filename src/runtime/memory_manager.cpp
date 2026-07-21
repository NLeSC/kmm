#include <chrono>
#include <cstring>

#include "fmt/format.h"
#include "spdlog/spdlog.h"

#include "kmm/runtime/memory_manager.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

struct BufferEntry {
    KMM_NOT_COPYABLE_OR_MOVABLE(BufferEntry)

  public:
    BufferEntry() = default;

    bool is_allocated = false;
    bool is_valid = false;

    size_t num_allocation_locks = 0;

    // let `<=` the happens-before operator. Then it should ALWAYS hold that
    // * epoch_event <= write_events
    // * write_events <= access_events
    DeviceEventSet epoch_event;
    DeviceEventSet write_events;
    DeviceEventSet access_events;
};

struct HostEntry: public BufferEntry {
    void* data = nullptr;
};

struct DeviceEntry: public BufferEntry {
    g_device_ptr_t data = 0;

    MemoryManager::Buffer* lru_older = nullptr;
    MemoryManager::Buffer* lru_newer = nullptr;
};

struct MemoryManager::Transaction {
    KMM_NOT_COPYABLE_OR_MOVABLE(Transaction);

  public:
    Transaction(uint64_t id, std::shared_ptr<Transaction> parent) : id(id), parent(parent) {}

    uint64_t id;
    std::shared_ptr<Transaction> parent;
    std::chrono::system_clock::time_point created_at = std::chrono::system_clock::now();
};

struct MemoryManager::Request {
    KMM_NOT_COPYABLE_OR_MOVABLE(Request);

  public:
    Request(
        uint64_t identifier,
        std::shared_ptr<Buffer> buffer,
        MemoryId memory_id,
        AccessMode mode,
        std::shared_ptr<Transaction> parent
    ) :
        identifier(identifier),
        buffer(std::move(buffer)),
        memory_id(memory_id),
        mode(mode),
        parent(std::move(parent)) {}

    enum struct Status { Init, Allocated, Locked, Ready, Deleted };
    Status status = Status::Init;
    uint64_t identifier;
    std::shared_ptr<Buffer> buffer;
    MemoryId memory_id;
    AccessMode mode;
    std::shared_ptr<Transaction> parent;
    std::chrono::system_clock::time_point created_at = std::chrono::system_clock::now();

    bool allocation_acquired = false;
    Request* allocation_next = nullptr;
    Request* allocation_prev = nullptr;

    bool access_acquired = false;
    Request* access_next = nullptr;
    Request* access_prev = nullptr;
};

struct MemoryManager::Buffer {
    KMM_NOT_COPYABLE_OR_MOVABLE(Buffer);

  public:
    Buffer(std::string name, BufferLayout layout) : name(std::move(name)), layout(layout) {
        if (this->name.empty()) {
            this->name = std::to_string(std::intptr_t(this));
        }
    }

    std::string name;
    BufferLayout layout;
    HostEntry host_entry;
    DeviceEntry device_entry[MAX_DEVICES];
    size_t num_requests_active = 0;
    bool is_deleted = false;

    Request* access_head = nullptr;
    Request* access_first_pending = nullptr;
    Request* access_tail = nullptr;

    KMM_INLINE BufferEntry& entry(MemoryId memory_id) noexcept {
        if (memory_id.is_host()) {
            return host_entry;
        } else {
            return device_entry[memory_id.as_device()];
        }
    }

    void add_to_access_queue(Request& req) noexcept {
        this->num_requests_active++;

        if (this->access_tail == nullptr) {
            this->access_head = &req;
            this->access_tail = &req;
        } else {
            auto* prev = this->access_tail;
            prev->access_next = &req;
            req.access_prev = prev;

            this->access_tail = &req;
        }

        if (this->access_first_pending == nullptr) {
            this->access_first_pending = &req;
        }
    }

    void remove_from_access_queue(Request& req) noexcept {
        this->num_requests_active--;
        auto* prev = std::exchange(req.access_prev, nullptr);
        auto* next = std::exchange(req.access_next, nullptr);

        if (prev != nullptr) {
            prev->access_next = next;
        } else {
            KMM_ASSERT(this->access_head == &req);
            this->access_head = next;
        }

        if (next != nullptr) {
            next->access_prev = prev;
        } else {
            KMM_ASSERT(this->access_tail == &req);
            this->access_tail = prev;
        }

        if (this->access_first_pending == &req) {
            this->access_first_pending = next;
        }

        if (req.access_acquired) {
            req.access_acquired = false;

            // Poll queue, releasing the lock might allow another request to gain access
            poll_access_queue();
        }
    }

    bool is_access_allowed(const Request& req) const noexcept {
        auto mode = req.mode;

        for (auto* it = this->access_head; it != this->access_first_pending; it = it->access_next) {
            // Two exclusive requests can never be granted access simultaneously
            if (mode == AccessMode::Exclusive || it->mode == AccessMode::Exclusive) {
                return false;
            }

            // Two non-read requests can only be granted simultaneously if operating on the same memory.
            if (mode != AccessMode::Read || it->mode != AccessMode::Read) {
                if (req.memory_id != it->memory_id) {
                    return false;
                }
            }
        }

        return true;
    }

    void poll_access_queue() noexcept {
        while (this->access_first_pending != nullptr) {
            auto* req = this->access_first_pending;

            if (!is_access_allowed(*req)) {
                return;
            }

            req->access_acquired = true;
            spdlog::trace(
                "access to buffer {} was granted to request {} (memory={}, mode={})",
                this->name,
                req->identifier,
                req->memory_id,
                req->mode
            );

            this->access_first_pending = this->access_first_pending->access_next;
        }
    }

    bool request_has_access(const Request& req) {
        poll_access_queue();
        return req.access_acquired;
    }
};

struct MemoryManager::Device {
    KMM_NOT_COPYABLE_OR_MOVABLE(Device);

  public:
    DeviceId device_id;
    bool printed_offload_warning = false;

    Buffer* lru_oldest = nullptr;
    Buffer* lru_newest = nullptr;

    Request* allocation_head = nullptr;
    Request* allocation_first_pending = nullptr;
    Request* allocation_tail = nullptr;

    Device(DeviceId id) : device_id(id) {}

    void add_to_allocation_queue(Request& req) noexcept {
        auto* tail = this->allocation_tail;

        if (tail == nullptr) {
            this->allocation_head = &req;
        } else {
            tail->allocation_next = &req;
            req.allocation_prev = tail;
        }

        this->allocation_tail = &req;

        if (this->allocation_first_pending == nullptr) {
            this->allocation_first_pending = &req;
        }
    }

    void remove_from_allocation_queue(Request& req) noexcept {
        auto* prev = std::exchange(req.allocation_prev, nullptr);
        auto* next = std::exchange(req.allocation_next, nullptr);

        if (prev != nullptr) {
            prev->allocation_next = next;
        } else {
            KMM_ASSERT(this->allocation_head == &req);
            this->allocation_head = next;
        }

        if (next != nullptr) {
            next->allocation_prev = prev;
        } else {
            KMM_ASSERT(this->allocation_tail == &req);
            this->allocation_tail = prev;
        }

        if (this->allocation_first_pending == &req) {
            this->allocation_first_pending = next;
        }
    }

    void add_to_lru(Buffer& buffer) noexcept {
        auto& device_entry = buffer.device_entry[device_id];
        auto& device = *this;

        KMM_ASSERT(device_entry.is_allocated);
        KMM_ASSERT(device_entry.num_allocation_locks == 0);

        auto* prev = device.lru_newest;
        if (prev != nullptr) {
            prev->device_entry[device_id].lru_newer = &buffer;
        } else {
            device.lru_oldest = &buffer;
        }

        device_entry.lru_older = prev;
        device_entry.lru_newer = nullptr;
        device.lru_newest = &buffer;
    }

    void remove_from_lru(Buffer& buffer) noexcept {
        auto& device_entry = buffer.device_entry[device_id];
        auto& device = *this;

        KMM_ASSERT(device_entry.is_allocated);
        KMM_ASSERT(device_entry.num_allocation_locks == 0);

        auto* prev = std::exchange(device_entry.lru_newer, nullptr);
        auto* next = std::exchange(device_entry.lru_older, nullptr);

        if (prev != nullptr) {
            prev->device_entry[device_id].lru_older = next;
        } else {
            device.lru_newest = next;
        }

        if (next != nullptr) {
            next->device_entry[device_id].lru_newer = prev;
        } else {
            device.lru_oldest = prev;
        }
    }

    void increment_allocation_locks(Buffer& buffer, Request& req) {
        KMM_ASSERT(req.allocation_acquired == false);
        KMM_ASSERT(allocation_first_pending == &req);

        auto& device_entry = buffer.device_entry[device_id];
        KMM_ASSERT(device_entry.is_allocated);

        if (device_entry.num_allocation_locks == 0) {
            remove_from_lru(buffer);
        }

        device_entry.num_allocation_locks++;

        req.allocation_acquired = true;
        this->allocation_first_pending = req.allocation_next;
    }

    void decrement_allocation_locks(Buffer& buffer) {
        auto& device_entry = buffer.device_entry[device_id];
        KMM_ASSERT(device_entry.is_allocated);
        KMM_ASSERT(device_entry.num_allocation_locks > 0);

        device_entry.num_allocation_locks--;

        if (device_entry.num_allocation_locks == 0) {
            add_to_lru(buffer);
        }
    }
};

template<size_t... Id>
static auto make_devices(std::index_sequence<Id...> /*unused*/) {
    return new MemoryManager::Device[MAX_DEVICES] {DeviceId(Id)...};
}

MemoryManager::MemoryManager(std::shared_ptr<MemorySystem> memory_system) :
    m_memory(std::move(memory_system)),
    m_devices(make_devices(std::make_index_sequence<MAX_DEVICES>())) {}

MemoryManager::~MemoryManager() {
    KMM_ASSERT(m_buffers.empty());
}

bool MemoryManager::is_idle(DeviceStreamManager& streams) const {
    bool result = true;

    for (const auto& buffer : m_buffers) {
        for (auto& e : buffer->device_entry) {
            result &= streams.is_ready(e.access_events);
            result &= streams.is_ready(e.write_events);
            result &= streams.is_ready(e.epoch_event);
        }

        auto& e = buffer->host_entry;
        result &= streams.is_ready(e.access_events);
        result &= streams.is_ready(e.write_events);
        result &= streams.is_ready(e.epoch_event);
    }

    return result;
}

std::shared_ptr<MemoryManager::Transaction> MemoryManager::create_transaction(
    std::shared_ptr<Transaction> parent
) {
    auto id = m_next_transaction_id++;
    return std::make_shared<Transaction>(id, parent);
}

std::shared_ptr<MemoryManager::Buffer> MemoryManager::create_buffer(
    BufferLayout layout,
    std::string name
) {
    // Size cannot be zero
    if (layout.size_in_bytes == 0) {
        layout.size_in_bytes = 1;
    }

    // Make sure alignment is power of two and size is multiple of alignment
    layout.alignment = round_up_to_power_of_two(layout.alignment);
    layout.size_in_bytes = round_up_to_multiple(layout.size_in_bytes, layout.alignment);

    auto buffer = std::make_shared<Buffer>(std::move(name), std::move(layout));
    m_buffers.emplace(buffer);
    return buffer;
}

void MemoryManager::delete_buffer(std::shared_ptr<Buffer> buffer) {
    if (buffer->is_deleted) {
        return;
    }

    KMM_ASSERT(buffer->num_requests_active == 0);
    KMM_ASSERT(buffer->access_head == nullptr);
    KMM_ASSERT(buffer->access_first_pending == nullptr);
    KMM_ASSERT(buffer->access_tail == nullptr);

    buffer->is_deleted = true;
    m_buffers.erase(buffer);

    deallocate_host(*buffer);

    for (size_t i = 0; i < MAX_DEVICES; i++) {
        deallocate_device_async(DeviceId(i), *buffer);
    }

    check_consistency();
}

std::shared_ptr<MemoryManager::Request> MemoryManager::create_request(
    std::shared_ptr<Buffer> buffer,
    MemoryId memory_id,
    AccessMode mode,
    std::shared_ptr<Transaction> parent
) {
    KMM_ASSERT(!buffer->is_deleted);
    auto req = std::make_shared<Request>(m_next_request_id++, buffer, memory_id, mode, parent);

    m_active_requests.insert(req);
    buffer->add_to_access_queue(*req);

    if (memory_id.is_device()) {
        device_at(memory_id.as_device()).add_to_allocation_queue(*req);
    }

    return req;
}

Poll MemoryManager::poll_request(Request& req, DeviceEventSet& deps_out) {
    auto& buffer = *req.buffer;
    auto memory_id = req.memory_id;

    if (req.status == Request::Status::Init) {
        if (memory_id.is_host()) {
            lock_allocation_host(buffer, memory_id.device_affinity(), req);
        } else {
            if (!try_lock_allocation_device(memory_id.as_device(), buffer, req)) {
                return Poll::Pending;
            }
        }

        req.status = Request::Status::Allocated;
    }

    if (req.status == Request::Status::Allocated) {
        if (!buffer.request_has_access(req)) {
            return Poll::Pending;
        }

        req.status = Request::Status::Locked;
    }

    if (req.status == Request::Status::Locked) {
        prepare_access_to_buffer(memory_id, buffer, req.mode, deps_out);
        req.status = Request::Status::Ready;
    }

    if (req.status == Request::Status::Ready) {
        return Poll::Ready;
    }

    throw std::runtime_error("cannot poll a deleted request");
}

void MemoryManager::release_request(std::shared_ptr<Request> req, DeviceEvent event) {
    auto memory_id = req->memory_id;
    auto& buffer = *req->buffer;
    auto status = std::exchange(req->status, Request::Status::Deleted);

    if (status == Request::Status::Ready) {
        status = Request::Status::Locked;
    }

    if (status == Request::Status::Locked) {
        spdlog::trace(
            "access to buffer {} was revoked from request {} (memory={}, mode={}, GPU event={})",
            buffer.name,
            req->identifier,
            req->memory_id,
            req->mode,
            event
        );

        finalize_access_to_buffer(memory_id, buffer, req->mode, event);
        status = Request::Status::Allocated;
    }

    if (status == Request::Status::Allocated) {
        if (memory_id.is_host()) {
            unlock_allocation_host(buffer, *req);
        } else {
            unlock_allocation_device(memory_id.as_device(), buffer, *req);
        }

        status = Request::Status::Init;
    }

    if (status == Request::Status::Init) {
        if (memory_id.is_device()) {
            device_at(memory_id.as_device()).remove_from_allocation_queue(*req);
        }

        buffer.remove_from_access_queue(*req);
        m_active_requests.erase(req);
    }
}

BufferAccessor MemoryManager::get_accessor(Request& req) {
    KMM_ASSERT(req.status == Request::Status::Ready);
    KMM_ASSERT(m_buffers.count(req.buffer) > 0);

    const auto& buffer = *req.buffer;
    void* address;

    if (req.memory_id.is_host()) {
        address = buffer.host_entry.data;
    } else {
        address = reinterpret_cast<void*>(buffer.device_entry[req.memory_id.as_device()].data);
    }

    return BufferAccessor {
        .memory_id = req.memory_id,
        .layout = buffer.layout,
        .is_writable = req.mode != AccessMode::Read,
        .address = address
    };
}

void MemoryManager::allocate_host(Buffer& buffer, DeviceId device_affinity) {
    auto& host_entry = buffer.host_entry;
    size_t size_in_bytes = buffer.layout.size_in_bytes;

    KMM_ASSERT(host_entry.is_allocated == false);
    KMM_ASSERT(host_entry.num_allocation_locks == 0);

    spdlog::trace("allocate {} bytes on host", size_in_bytes, buffer.name);

    void* ptr;
    DeviceEventSet events;
    AllocationResult result = m_memory->allocate_host(size_in_bytes, device_affinity, &ptr, events);

    if (result != AllocationResult::Success) {
        throw std::runtime_error("could not allocate, out of host memory");
    }

    host_entry.data = ptr;
    host_entry.is_allocated = true;
    host_entry.is_valid = false;
    host_entry.epoch_event = events;
    host_entry.access_events = events;
    host_entry.write_events = std::move(events);
}

void MemoryManager::deallocate_host(Buffer& buffer) {
    auto& host_entry = buffer.host_entry;
    if (!host_entry.is_allocated) {
        return;
    }

    KMM_ASSERT(host_entry.num_allocation_locks == 0);
    KMM_ASSERT(buffer.access_head == nullptr);
    KMM_ASSERT(buffer.access_first_pending == nullptr);
    KMM_ASSERT(buffer.access_tail == nullptr);

    size_t size_in_bytes = buffer.layout.size_in_bytes;
    spdlog::trace(
        "free {} bytes on host (dependencies={})",
        size_in_bytes,
        buffer.name,
        host_entry.access_events
    );

    m_memory->deallocate_host(host_entry.data, size_in_bytes, std::move(host_entry.access_events));

    host_entry.data = nullptr;
    host_entry.is_allocated = false;
    host_entry.is_valid = false;
    host_entry.epoch_event.clear();
    host_entry.write_events.clear();
    host_entry.access_events.clear();
    host_entry.data = nullptr;
}

bool MemoryManager::try_free_device_memory(DeviceId device_id) {
    auto& device = device_at(device_id);
    auto* victim = device.lru_oldest;

    if (victim == nullptr) {
        return false;
    }

    if (victim->device_entry[device_id].is_valid) {
        bool valid_anywhere = victim->host_entry.is_valid;

        for (size_t i = 0; i < MAX_DEVICES; i++) {
            if (i != device_id) {
                valid_anywhere |= victim->device_entry[i].is_valid;
            }
        }

        if (!valid_anywhere) {
            if (!device.printed_offload_warning) {
                device.printed_offload_warning = true;
                spdlog::warn(
                    "GPU {} is out of memory. The system will now offload data from GPU memory to "
                    "host memory as a fallback, which may significantly reduce performance.",
                    device_id
                );
            }

            if (!victim->host_entry.is_allocated) {
                allocate_host(*victim, device_id);
            }

            copy_d2h(device_id, *victim);
        }
    }

    spdlog::debug(
        "evict buffer {} from GPU {}, frees {} bytes",
        victim->name,
        device_id,
        victim->layout.size_in_bytes
    );

    deallocate_device_async(device_id, *victim);
    return true;
}

AllocationResult MemoryManager::try_allocate_device_async(DeviceId device_id, Buffer& buffer) {
    auto& device_entry = buffer.device_entry[device_id];

    if (device_entry.is_allocated) {
        return AllocationResult::Success;
    }

    KMM_ASSERT(device_entry.num_allocation_locks == 0);
    spdlog::trace(
        "allocate {} bytes on GPU {} for buffer {}",
        buffer.layout.size_in_bytes,
        device_id,
        buffer.name
    );

    g_device_ptr_t ptr_out;
    DeviceEventSet events;
    auto result =
        m_memory->allocate_device(device_id, buffer.layout.size_in_bytes, &ptr_out, events);

    if (result != AllocationResult::Success) {
        return result;
    }

    device_entry.data = ptr_out;
    device_entry.is_allocated = true;
    device_entry.is_valid = false;
    device_entry.epoch_event = events;
    device_entry.access_events = events;
    device_entry.write_events = std::move(events);

    device_at(device_id).add_to_lru(buffer);
    check_consistency();
    return result;
}

void MemoryManager::deallocate_device_async(DeviceId device_id, Buffer& buffer) {
    auto& device_entry = buffer.device_entry[device_id];
    KMM_ASSERT(device_entry.num_allocation_locks == 0);

    if (!device_entry.is_allocated) {
        return;
    }

    size_t size_in_bytes = buffer.layout.size_in_bytes;
    spdlog::trace(
        "free {} bytes for buffer {} on GPU {} (dependencies={})",
        size_in_bytes,
        buffer.name,
        device_id,
        device_entry.access_events
    );

    m_memory->deallocate_device(
        device_id,
        device_entry.data,
        size_in_bytes,
        std::move(device_entry.access_events)
    );

    device_at(device_id).remove_from_lru(buffer);

    device_entry.is_allocated = false;
    device_entry.is_valid = false;
    device_entry.epoch_event.clear();
    device_entry.write_events.clear();
    device_entry.access_events.clear();
    device_entry.data = 0;

    check_consistency();
}

void MemoryManager::lock_allocation_host(Buffer& buffer, DeviceId device_affinity, Request& req) {
    auto& host_entry = buffer.host_entry;

    if (!host_entry.is_allocated) {
        allocate_host(buffer, device_affinity);
    }

    host_entry.num_allocation_locks++;
    spdlog::trace(
        "lock allocation on host of buffer {} for request {}",
        buffer.name,
        req.identifier
    );
}

void MemoryManager::unlock_allocation_host(Buffer& buffer, Request& req) {
    auto& host_entry = buffer.host_entry;

    KMM_ASSERT(host_entry.is_allocated);
    KMM_ASSERT(host_entry.num_allocation_locks > 0);

    host_entry.num_allocation_locks--;
    spdlog::trace(
        "unlock allocation on host of buffer {} for request {}",
        buffer.name,
        req.identifier
    );
}

bool MemoryManager::try_lock_allocation_device(DeviceId device_id, Buffer& buffer, Request& req) {
    KMM_ASSERT(req.allocation_acquired == false);

    if (device_at(device_id).allocation_first_pending != &req) {
        return false;
    }

    while (true) {
        // Try to allocate
        auto result = try_allocate_device_async(device_id, buffer);

        if (result == AllocationResult::Success) {
            break;
        }

        // No memory available, try to free memory
        if (try_free_device_memory(device_id)) {
            continue;
        }

        if (!is_out_of_memory(device_id, req)) {
            return false;
        }

        throw std::runtime_error(fmt::format(
            "cannot allocate {} bytes on GPU {}, out of memory",
            buffer.layout.size_in_bytes,
            device_id
        ));
    }

    spdlog::trace(
        "lock allocation on GPU {} of buffer {} for request {}",
        device_id,
        buffer.name,
        req.identifier
    );

    device_at(device_id).increment_allocation_locks(buffer, req);
    return true;
}

void MemoryManager::unlock_allocation_device(
    DeviceId device_id,
    Buffer& buffer,
    Request& req
) noexcept {
    spdlog::trace(
        "unlock allocation on GPU {} of buffer {} for request {}",
        device_id,
        buffer.name,
        req.identifier
    );

    device_at(device_id).decrement_allocation_locks(buffer);
    check_consistency();
}

void MemoryManager::prepare_access_to_buffer(
    MemoryId memory_id,
    Buffer& buffer,
    AccessMode mode,
    DeviceEventSet& deps_out
) {
    bool is_writer = mode != AccessMode::Read;
    bool is_exclusive = mode == AccessMode::Exclusive;
    auto& entry = buffer.entry(memory_id);

    make_entry_valid(memory_id, buffer, deps_out);

    if (is_writer) {
        make_entry_exclusive(memory_id, buffer, deps_out);
    }

    if (is_exclusive) {
        deps_out.insert(entry.access_events);
    }
}

void MemoryManager::finalize_access_to_buffer(
    MemoryId memory_id,
    Buffer& buffer,
    AccessMode mode,
    DeviceEvent event
) noexcept {
    bool is_writer = mode != AccessMode::Read;
    bool is_exclusive = mode == AccessMode::Exclusive;
    auto& entry = buffer.entry(memory_id);

    entry.access_events.insert(event);

    if (is_writer) {
        entry.write_events.insert(event);
    }

    if (is_exclusive) {
        entry.epoch_event.insert(event);
    }
}

std::optional<DeviceId> MemoryManager::find_valid_device_entry(const Buffer& buffer) {
    for (size_t device_id = 0; device_id < MAX_DEVICES; device_id++) {
        if (buffer.device_entry[device_id].is_valid) {
            return DeviceId(device_id);
        }
    }

    return std::nullopt;
}

void MemoryManager::make_entry_valid(MemoryId memory_id, Buffer& buffer, DeviceEventSet& deps_out) {
    auto& entry = buffer.entry(memory_id);

    KMM_ASSERT(entry.is_allocated);
    deps_out.insert(entry.epoch_event);

    if (entry.is_valid) {
        return;
    }

    if (memory_id.is_host()) {
        if (auto src_id = find_valid_device_entry(buffer)) {
            deps_out.insert(copy_d2h(*src_id, buffer));
            return;
        }
    } else {
        auto device_id = memory_id.as_device();

        if (buffer.host_entry.is_valid) {
            deps_out.insert(copy_h2d(device_id, buffer));
            return;
        }

        if (auto src_id = find_valid_device_entry(buffer)) {
            if (m_memory->is_copy_supported(*src_id, memory_id)) {
                deps_out.insert(copy_d2d(*src_id, device_id, buffer));
            } else {
                if (!buffer.host_entry.is_allocated) {
                    allocate_host(buffer, *src_id);
                }
                deps_out.insert(copy_d2h(*src_id, buffer));
                deps_out.insert(copy_h2d(device_id, buffer));
            }

            return;
        }
    }

    entry.is_valid = true;
}

void MemoryManager::make_entry_exclusive(
    MemoryId memory_id,
    Buffer& buffer,
    DeviceEventSet& deps_out
) {
    make_entry_valid(memory_id, buffer, deps_out);

    // invalidate host if necessary
    if (memory_id != MemoryId::host()) {
        buffer.host_entry.is_valid = false;
        deps_out.insert(buffer.host_entry.access_events);
    }

    // Invalidate all _other_ device entries
    for (size_t i = 0; i < MAX_DEVICES; i++) {
        if (memory_id == MemoryId(DeviceId(i))) {
            continue;
        }

        auto& peer_entry = buffer.device_entry[i];
        peer_entry.is_valid = false;
        deps_out.insert(peer_entry.access_events);
    }
}

DeviceEvent MemoryManager::copy_h2d(DeviceId device_id, Buffer& buffer) {
    spdlog::trace(
        "copy {} bytes from host to GPU {} for buffer {}",
        buffer.layout.size_in_bytes,
        device_id,
        buffer.name
    );

    auto& host_entry = buffer.host_entry;
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(host_entry.is_allocated && device_entry.is_allocated);
    KMM_ASSERT(host_entry.is_valid && !device_entry.is_valid);

    DeviceEventSet deps = device_entry.access_events | host_entry.write_events;

    auto event = m_memory->copy_host_to_device(
        device_id,
        host_entry.data,
        device_entry.data,
        buffer.layout.size_in_bytes,
        std::move(deps)
    );

    host_entry.access_events.insert(event);
    device_entry.epoch_event = {event};
    device_entry.access_events = {event};
    device_entry.write_events = {event};

    device_entry.is_valid = true;
    return event;
}

DeviceEvent MemoryManager::copy_d2h(DeviceId device_id, Buffer& buffer) {
    spdlog::trace(
        "copy {} bytes from GPU {} to host for buffer {}",
        buffer.layout.size_in_bytes,
        device_id,
        buffer.name
    );

    auto& host_entry = buffer.host_entry;
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(host_entry.is_allocated && device_entry.is_allocated);
    KMM_ASSERT(!host_entry.is_valid && device_entry.is_valid);

    DeviceEventSet deps = device_entry.write_events | host_entry.access_events;

    auto event = m_memory->copy_device_to_host(
        device_id,
        device_entry.data,
        host_entry.data,
        buffer.layout.size_in_bytes,
        std::move(deps)
    );

    device_entry.access_events.insert(event);
    host_entry.epoch_event.insert(event);
    host_entry.access_events.insert(event);
    host_entry.write_events.insert(event);
    host_entry.is_valid = true;

    return event;
}

DeviceEvent MemoryManager::copy_d2d(
    DeviceId device_src_id,
    DeviceId device_dst_id,
    Buffer& buffer
) {
    spdlog::trace(
        "copy {} bytes from GPU {} to GPU {} for buffer {}",
        buffer.layout.size_in_bytes,
        device_src_id,
        device_dst_id,
        buffer.name
    );

    auto& src_entry = buffer.device_entry[device_src_id];
    auto& dst_entry = buffer.device_entry[device_dst_id];

    KMM_ASSERT(src_entry.is_allocated && dst_entry.is_allocated);
    KMM_ASSERT(src_entry.is_valid && !dst_entry.is_valid);

    DeviceEventSet deps = dst_entry.access_events | src_entry.write_events;

    auto event = m_memory->copy_device_to_device(
        device_src_id,
        device_dst_id,
        src_entry.data,
        dst_entry.data,
        buffer.layout.size_in_bytes,
        std::move(deps)
    );

    src_entry.access_events.insert(event);
    dst_entry.epoch_event = {event};
    dst_entry.access_events = {event};
    dst_entry.write_events = {event};
    dst_entry.is_valid = true;

    return event;
}

MemoryManager::Device& MemoryManager::device_at(DeviceId id) noexcept {
    KMM_ASSERT(id < MAX_DEVICES);
    return m_devices[id];
}

bool MemoryManager::is_out_of_memory(DeviceId device_id, Request& req) {
    std::unordered_set<const Transaction*> waiting_transactions;
    auto& device = device_at(device_id);

    // First, iterate over the requests that are waiting for allocation. Mark all the
    // related transactions as `waiting` by adding them to `waiting_transactions`
    for (auto* it = device.allocation_first_pending; it != nullptr; it = it->allocation_next) {
        auto* p = it->parent.get();

        while (p != nullptr) {
            waiting_transactions.insert(p);
            p = p->parent.get();
        }
    }

    // Next, iterate over the requests that have been granted an allocation. If the associated
    // transaction of one of the requests has not been marked as waiting, we are not out of memory
    // since that transaction will release its memory again at some point in the future.
    for (auto* it = device.allocation_head; it != device.allocation_first_pending;
         it = it->allocation_next) {
        auto* p = it->parent.get();

        if (waiting_transactions.find(p) == waiting_transactions.end()) {
            return false;
        }
    }

    spdlog::error(
        "out of memory for GPU {}, failed to allocate {} bytes for request {} of buffer {}",
        device_id,
        req.buffer->layout.size_in_bytes,
        req.buffer->name,
        req.identifier
    );

    spdlog::error("following buffers are currently allocated: ");

    for (const auto& buffer : m_buffers) {
        auto& entry = buffer->entry(device_id);

        if (entry.is_allocated) {
            spdlog::error(
                " - buffer {} ({} bytes, {} requests active, {} allocation locks)",
                buffer->name,
                buffer->layout.size_in_bytes,
                buffer->num_requests_active,
                entry.num_allocation_locks
            );
        }
    }

    spdlog::error("following requests have been granted:");
    for (auto* it = device.allocation_head; it != device.allocation_first_pending;
         it = it->allocation_next) {
        spdlog::error(
            " - request {} ({} bytes, buffer {}, transaction {})",
            it->identifier,
            it->buffer->layout.size_in_bytes,
            it->buffer->name,
            it->parent->id
        );
    }

    spdlog::error("following requests are pending:");
    for (auto* it = device.allocation_first_pending; it != nullptr; it = it->allocation_next) {
        spdlog::error(
            " - request {} ({} bytes, buffer {}, transaction {})",
            it->identifier,
            it->buffer->layout.size_in_bytes,
            it->buffer->name,
            it->parent->id
        );
    }

    return true;
}

void MemoryManager::check_consistency() const {}

// This is here to check the consistency of the data structures while debugging.
/*
void MemoryManager::check_consistency() const {
    for (size_t i = 0; i < MAX_DEVICES; i++) {
        std::unordered_set<Buffer*> available_buffers;
        auto id = DeviceId(i);
        auto& device = m_devices[i];

        auto* prev = (Buffer*) nullptr;
        auto* current = device.lru_oldest;

        while (current != nullptr) {
            available_buffers.insert(current);
            auto& entry = current->device_entry[id];

            KMM_ASSERT(entry.num_allocation_locks == 0);
            KMM_ASSERT(entry.lru_older == prev);

            prev = current;
            current = entry.lru_newer;
        }

        KMM_ASSERT(prev == device.lru_newest);

        for (const auto& buffer: m_buffers) {
            auto& entry = buffer->device_entry[id];

            if (entry.is_allocated && entry.num_allocation_locks == 0) {
                KMM_ASSERT(available_buffers.find(buffer.get()) != available_buffers.end());
            }
        }
    }
}
*/

}  // namespace kmm
