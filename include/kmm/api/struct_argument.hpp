#pragma once

#include "kmm/api/argument.hpp"

namespace kmm {

template<typename Type, typename... Fields>
struct StructArgument {
    static constexpr size_t num_fields = sizeof...(Fields);
    std::tuple<Fields...> fields;
};

template<typename Type, typename... Fields>
struct StructArgumentHandler {
    using type = StructArgument<Type, typename ArgumentHandler<Fields>::type...>;

    StructArgumentHandler(const Type&, Fields... fields) : m_handlers(fields...) {}

    void initialize(const TaskGroupInit& init) {
        initialize_impl(init, std::make_index_sequence<type::num_fields>());
    }

    type before_submit(TaskInstance& task) {
        return before_submit_impl(task, std::make_index_sequence<type::num_fields>());
    }

    void after_submit(const TaskSubmissionResult& result) {
        after_submit_impl(result, std::make_index_sequence<type::num_fields>());
    }

    void commit(const TaskGroupCommit& commit) {
        commit_impl(commit, std::make_index_sequence<type::num_fields>());
    }

  private:
    template<size_t... Is>
    void initialize_impl(const TaskGroupInit& init, std::index_sequence<Is...>) {
        (std::get<Is>(m_handlers).initialize(init), ...);
    }

    template<size_t... Is>
    type before_submit_impl(TaskInstance& task, std::index_sequence<Is...>) {
        return {.fields = {(std::get<Is>(m_handlers).before_submit(task))...}};
    }

    template<size_t... Is>
    void after_submit_impl(const TaskSubmissionResult& result, std::index_sequence<Is...>) {
        (std::get<Is>(m_handlers).after_submit(result), ...);
    }

    template<size_t... Is>
    void commit_impl(const TaskGroupCommit& commit, std::index_sequence<Is...>) {
        (std::get<Is>(m_handlers).commit(commit), ...);
    }

    std::tuple<ArgumentHandler<Fields>...> m_handlers;
};

template<ExecutionSpace Space, typename View, typename Type, typename... Fields>
struct StructArgumentUnpack {
    static View call(TaskContext& context, StructArgument<Type, Fields...>& data) {
        return call_impl(context, data, std::index_sequence_for<Fields...>());
    }

  private:
    template<size_t... Is>
    static View
    call_impl(TaskContext& context, StructArgument<Type, Fields...>& data, std::index_sequence<Is...>) {
        return {ArgumentUnpack<Space, Fields>::call(context, std::get<Is>(data.fields))...};
    }
};

}  // namespace kmm

#define KMM_DEFINE_STRUCT_ARGUMENT_IMPL(UNIQUE_NAME, T, ...)                         \
    static auto UNIQUE_NAME(const T& it) {                                           \
        return kmm::StructArgumentHandler(it, __VA_ARGS__);                          \
    }                                                                                \
    template<>                                                                       \
    struct kmm::ArgumentHandler<T>: decltype(UNIQUE_NAME(std::declval<T>())) {       \
        ArgumentHandler(const T& it) : decltype(UNIQUE_NAME(it))(it, __VA_ARGS__) {} \
    };

#define KMM_DEFINE_STRUCT_ARGUMENT(T, ...)                        \
    KMM_DEFINE_STRUCT_ARGUMENT_IMPL(                              \
        KMM_CONCAT(__kmm_argument_handle_type_helper_, __LINE__), \
        T,                                                        \
        __VA_ARGS__                                               \
    )

#define KMM_DEFINE_STRUCT_VIEW(T, V)                                      \
    template<kmm::ExecutionSpace Space, typename... Fields>               \
    struct kmm::ArgumentUnpack<Space, kmm::StructArgument<T, Fields...>>: \
        kmm::StructArgumentUnpack<Space, V, T, Fields...> {};
