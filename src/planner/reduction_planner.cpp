#include "kmm/planner/reduction_planner.hpp"
#include "kmm/runtime/task_graph.hpp"

namespace kmm {

template<size_t N>
ArrayReductionPlanner<N>::ArrayReductionPlanner(
    std::shared_ptr<ArrayDescriptor<N>> instance,
    Reduction op
) :
    m_lock(instance->m_mutex, std::try_to_lock),
    m_instance(std::move(instance)),
    m_reduction(op) {
    KMM_ASSERT(m_instance);

    if (!m_lock) {
        throw std::runtime_error(
            "array could not be locked for reductions, which may happen if "
            "the same array is provided multiple times as an argument to a kernel"
        );
    }
}

template<size_t N>
ArrayReductionPlanner<N>::~ArrayReductionPlanner() {}

template<size_t N>
BufferRequirement ArrayReductionPlanner<N>::prepare_access(
    TaskGraph& stage,
    MemoryId memory_id,
    Bounds<N>& region,
    size_t replication_factor,
    EventList& deps_out
) {
    size_t chunk_index = m_instance->m_distribution.region_to_chunk_index(region);
    auto chunk = m_instance->m_distribution.chunk(chunk_index);

    region = Bounds<N>::from_offset_size(chunk.offset, chunk.size);

    auto dtype = m_instance->data_type();
    auto num_elements = checked_mul(checked_cast<size_t>(region.volume()), replication_factor);
    BufferLayout layout = BufferLayout::for_type(dtype, num_elements);

    auto buffer_id = stage.create_buffer(layout);

    auto fill_event = stage.insert_node(CommandFill {
        .dst_buffer = buffer_id,
        .memory_id = memory_id,
        .definition = FillDef(
            dtype.size_in_bytes(),
            num_elements,
            reduction_identity_value(dtype, m_reduction).data()
        )
    });

    m_partial_buffers.push_back(PartialReductionBuffer {
        .chunk_index = chunk_index,
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .replication_factor = replication_factor,
        .creation_event = fill_event,
        .write_events = {}
    });

    deps_out.push_back(fill_event);

    return BufferRequirement {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .access_mode = AccessMode::Exclusive
    };
}

template<size_t N>
void ArrayReductionPlanner<N>::finalize_access(TaskGraph& stage, EventId event_id) {
    KMM_ASSERT(!m_partial_buffers.empty());
    m_partial_buffers.back().write_events.push_back(event_id);
}

template<size_t N>
std::pair<BufferId, EventId> ArrayReductionPlanner<N>::reduce_per_chunk_and_memory(
    TaskGraph& stage,
    size_t chunk_index,
    MemoryId memory_id,
    PartialReductionBuffer** buffers,
    size_t num_buffers
) {
    auto dtype = m_instance->data_type();
    auto chunk = m_instance->distribution().chunk(chunk_index);
    auto num_elements = checked_cast<size_t>(chunk.size.volume());
    auto layout = BufferLayout::for_type(dtype, num_elements);

    auto scratch_buffer = stage.create_buffer(layout.repeat(num_buffers));
    auto scratch_writes = EventList {};

    for (size_t i = 0; i < num_buffers; i++) {
        EventId event_id = stage.insert_node(
            CommandReduction {
                buffers[i]->buffer_id,
                scratch_buffer,
                memory_id,
                ReductionDef {
                    .operation = m_reduction,
                    .data_type = dtype,
                    .num_outputs = num_elements,
                    .num_inputs_per_output = buffers[i]->replication_factor,
                    .output_offset_elements = i * num_elements
                },
            },
            std::move(buffers[i]->write_events)
        );

        stage.delete_buffer(buffers[i]->buffer_id, {event_id});
        scratch_writes.push_back(event_id);
    }

    BufferId final_buffer = stage.create_buffer(layout);
    EventId final_write = stage.insert_node(
        CommandReduction {
            scratch_buffer,
            final_buffer,
            memory_id,
            ReductionDef {
                .operation = m_reduction,
                .data_type = dtype,
                .num_outputs = num_elements,
                .num_inputs_per_output = num_buffers,
            },
        },
        std::move(scratch_writes)
    );

    stage.delete_buffer(scratch_buffer, {final_write});
    return {final_buffer, final_write};
}

template<size_t N>
EventId ArrayReductionPlanner<N>::reduce_per_chunk(
    TaskGraph& stage,
    size_t chunk_index,
    PartialReductionBuffer** buffers,
    size_t num_buffers
) {
    auto chunk = m_instance->distribution().chunk(chunk_index);
    auto num_elements = checked_cast<size_t>(chunk.size.volume());
    auto dtype = m_instance->data_type();
    auto layout = BufferLayout::for_type(dtype, num_elements);

    std::sort(buffers, buffers + num_buffers, [&](const auto* a, const auto* b) {
        return a->memory_id < b->memory_id;
    });

    std::vector<std::tuple<MemoryId, BufferId, EventId>> intermediates;

    for (size_t begin = 0, end = begin; begin < num_buffers; begin = end) {
        auto memory_id = buffers[begin]->memory_id;

        while (end < num_buffers && buffers[end]->memory_id == memory_id) {
            end++;
        }

        auto [buffer_id, event_id] = reduce_per_chunk_and_memory(
            stage,
            chunk_index,
            memory_id,
            &buffers[begin],
            end - begin
        );

        intermediates.emplace_back(memory_id, buffer_id, event_id);
    }

    auto collect_buffer = stage.create_buffer(layout.repeat(intermediates.size()));
    auto collect_events = EventList {};

    for (size_t i = 0; i < intermediates.size(); i++) {
        auto [memory_id, buffer_id, event_id] = intermediates[i];

        auto element_size = dtype.size_in_bytes();
        auto copy_definition = CopyDef(element_size);
        copy_definition.add_dimension(
            num_elements,  //
            0,
            i * num_elements,
            element_size,
            element_size
        );

        auto copy_event = stage.insert_node(
            CommandCopy {
                .src_buffer = buffer_id,  //
                .src_memory = memory_id,
                .dst_buffer = collect_buffer,
                .dst_memory = chunk.owner_id,
                .definition = copy_definition
            },
            {event_id}
        );

        collect_events.push_back(copy_event);
        stage.delete_buffer(buffer_id, {copy_event});
    }

    auto& final_buffer = m_instance->m_buffers[chunk_index];
    collect_events.insert_all(final_buffer.last_access_events);

    auto final_event = stage.insert_node(
        CommandReduction {
            .src_buffer = collect_buffer,
            .dst_buffer = final_buffer.id,
            .memory_id = chunk.owner_id,
            .definition =
                ReductionDef {
                    .operation = m_reduction,
                    .data_type = m_instance->data_type(),
                    .num_outputs = num_elements,
                    .num_inputs_per_output = intermediates.size()
                }
        },
        std::move(collect_events)
    );

    stage.delete_buffer(collect_buffer, {final_event});

    final_buffer.last_write_event = final_event;
    final_buffer.last_access_events = {final_event};
    return final_event;
}

template<size_t N>
void ArrayReductionPlanner<N>::commit(TaskGraph& stage) {
    auto buffers = std::vector<PartialReductionBuffer*>();
    for (size_t i = 0; i < m_partial_buffers.size(); i++) {
        buffers.push_back(&m_partial_buffers[i]);
    }

    std::sort(buffers.begin(), buffers.end(), [&](const auto* a, const auto* b) {
        return a->chunk_index < b->chunk_index;
    });

    for (size_t begin = 0, end = begin; begin < buffers.size(); begin = end) {
        size_t chunk_index = buffers[begin]->chunk_index;

        while (end < buffers.size() && chunk_index == buffers[end]->chunk_index) {
            end++;
        }

        reduce_per_chunk(stage, chunk_index, &buffers[begin], end - begin);
    }

    m_partial_buffers.clear();
}

KMM_INSTANTIATE_ARRAY_IMPL(ArrayReductionPlanner)

}  // namespace kmm