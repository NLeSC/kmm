#include "kmm/core/panic.hpp"
#include "kmm/runtime/memops/types.hpp"

namespace kmm {

size_t data_type_size(DataType dtype) {
    switch (dtype) {
        case DataType::Unknown:
            break;
        case DataType::Int8:
        case DataType::Uint8:
            return 1;
        case DataType::Int16:
        case DataType::Uint16:
            return 2;
        case DataType::Int32:
        case DataType::Uint32:
        case DataType::Float32:
            return 4;
        case DataType::Int64:
        case DataType::Uint64:
        case DataType::Float64:
            return 8;
    }

    KMM_PANIC("invalid data type");
}

const char* data_type_name(DataType dtype) {
    switch (dtype) {
        case DataType::Unknown:
            break;
        case DataType::Int8:
            return "Int8";
        case DataType::Int16:
            return "Int16";
        case DataType::Int32:
            return "Int32";
        case DataType::Int64:
            return "Int64";
        case DataType::Uint8:
            return "Uint8";
        case DataType::Uint16:
            return "Uint16";
        case DataType::Uint32:
            return "Uint32";
        case DataType::Uint64:
            return "Uint64";
        case DataType::Float32:
            return "Float32";
        case DataType::Float64:
            return "Float64";
    }

    KMM_PANIC("invalid data type");
}

const char* reduction_op_name(ReductionOp op) {
    switch (op) {
        case ReductionOp::Sum:
            return "Sum";
        case ReductionOp::Product:
            return "Product";
        case ReductionOp::Min:
            return "Min";
        case ReductionOp::Max:
            return "Max";
    }

    KMM_PANIC("invalid reduction operator");
}

}  // namespace kmm
