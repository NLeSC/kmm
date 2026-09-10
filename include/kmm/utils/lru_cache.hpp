#pragma once

#include <ankerl/unordered_dense.h>
#include <list>
#include <utility>

namespace kmm {

/// \addtogroup utility
/// @{

/// A cache mapping keys to values that tracks usage order, so the caller can find and evict
/// the least-recently-used entry. `find`, `touch`, and `insert` all mark the given key as most
/// recently used. Eviction is not automatic; call `least_recently_used` and `remove` to evict.
template<typename K, typename V, typename Hash = ankerl::unordered_dense::hash<K>>
class lru_cache {
    struct entry {
        K key;
        V value;
    };

    using list_type = std::list<entry>;
    using map_type = ankerl::unordered_dense::map<K, typename list_type::iterator, Hash>;

  public:
    size_t size() const noexcept {
        return m_map.size();
    }

    bool is_empty() const noexcept {
        return m_map.empty();
    }

    bool contains(const K& key) const {
        return m_map.find(key) != m_map.end();
    }

    /// Returns a pointer to the value associated with `key`, marking it as most recently used,
    /// or `nullptr` if `key` is not present.
    V* find(const K& key) {
        auto it = m_map.find(key);

        if (it == m_map.end()) {
            return nullptr;
        }

        move_to_front(it->second);
        return &it->second->value;
    }

    const V* find(const K& key) const {
        return const_cast<lru_cache*>(this)->find(key);
    }

    /// Marks `key` as most recently used. Does nothing if `key` is not present.
    void touch(const K& key) {
        auto it = m_map.find(key);

        if (it != m_map.end()) {
            move_to_front(it->second);
        }
    }

    /// Inserts or overwrites the value for `key`, marking it as most recently used.
    void insert(K key, V value) {
        auto it = m_map.find(key);

        if (it != m_map.end()) {
            it->second->value = std::move(value);
            move_to_front(it->second);
            return;
        }

        m_list.push_front(entry {key, std::move(value)});
        m_map.emplace(std::move(key), m_list.begin());
    }

    /// Removes `key` from the cache. Returns `true` if `key` was present.
    bool remove(const K& key) {
        auto it = m_map.find(key);

        if (it == m_map.end()) {
            return false;
        }

        m_list.erase(it->second);
        m_map.erase(it);
        return true;
    }

    /// Returns the key of the least-recently-used entry, or `nullptr` if the cache is empty.
    const K* least_recently_used() const noexcept {
        if (m_list.empty()) {
            return nullptr;
        }

        return &m_list.back().key;
    }

    void clear() noexcept {
        m_list.clear();
        m_map.clear();
    }

  private:
    void move_to_front(typename list_type::iterator it) {
        m_list.splice(m_list.begin(), m_list, it);
    }

    list_type m_list;
    map_type m_map;
};

/// @}

}  // namespace kmm
