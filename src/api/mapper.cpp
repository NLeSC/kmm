#include <cstdint>
#include <numeric>

#include "kmm/api/mapper.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

IndexMap::IndexMap(Axis variable, int64_t scale, int64_t offset, int64_t length, int64_t divisor) :
    m_axis(variable),
    m_scale(scale),
    m_offset(offset),
    m_length(length),
    m_divisor(divisor) {
    m_length = std::max<int64_t>(m_length, 0);

    if (m_divisor < 0) {
        m_scale = -m_scale;
        m_offset = -checked_add(m_offset, m_length - 1);
        m_divisor = -m_divisor;
    }

    if (m_scale != 1 && m_divisor != 1) {
        auto common = std::gcd(std::gcd(m_scale, m_offset), std::gcd(m_length, m_divisor));

        if (common != 1) {
            m_scale /= common;
            m_offset /= common;
            m_length /= common;
            m_divisor /= common;
        }
    }
}

IndexMap IndexMap::range(IndexMap begin, IndexMap end) {
    if (begin.m_scale != end.m_scale || begin.m_divisor != end.m_divisor) {
        throw std::runtime_error(fmt::format(
            "`range` requires two expressions having the same scaling factor, given: `{}` and `{}`",
            begin,
            end
        ));
    }

    if (begin.m_axis.get() != end.m_axis.get() && begin.m_scale != 0) {
        throw std::runtime_error(fmt::format(
            "`range` requires two expression operating on the same axis, given: `{}` and `{}`",
            begin,
            end
        ));
    }

    return {
        begin.m_axis,
        begin.m_scale,
        begin.m_offset,
        (end.m_offset - begin.m_offset) + end.m_length,
        begin.m_divisor
    };
}

IndexMap IndexMap::offset_by(int64_t offset) const {
    auto new_offset = checked_add(m_offset, checked_mul(m_divisor, offset));
    return {m_axis, m_scale, new_offset, m_length, m_divisor};
}

IndexMap IndexMap::scale_by(int64_t factor) const {
    if (factor < 0) {
        return negate().scale_by(-factor);
    }

    return {
        m_axis,
        checked_mul(m_scale, factor),
        checked_mul(m_offset, factor),
        checked_mul(m_length - 1, factor) + 1,
        m_divisor
    };
}

IndexMap IndexMap::divide_by(int64_t divisor) const {
    if (divisor < 0) {
        return negate().divide_by(-divisor);
    }

    return {m_axis, m_scale, m_offset, m_length, checked_mul(m_divisor, divisor)};
}

IndexMap IndexMap::negate() const {
    return {m_axis, -m_scale, -checked_add(m_offset, m_length - 1), m_length, m_divisor};
}

Bounds<1> IndexMap::apply(DomainChunk chunk) const {
    int64_t a0 = chunk.offset.get_or_default(m_axis.get());
    int64_t an = chunk.size.get_or_default(m_axis.get());

    int64_t b0;
    int64_t bn;

    if (m_scale == 0) {
        b0 = m_offset;
        bn = m_length;
    } else if (m_length <= 0 || an <= 0) {
        b0 = 0;
        bn = 0;
    } else if (m_scale > 0) {
        b0 = m_scale * a0 + m_offset;
        bn = m_scale * (an - 1) + m_length;
    } else {
        b0 = m_scale * (a0 + an - 1) + m_offset;
        bn = -m_scale * (an - 1) + m_length;
    }

    if (m_divisor > 1) {
        int64_t remainder = b0 % m_divisor;
        b0 = b0 / m_divisor;
        bn = (bn + remainder + m_divisor - 1) / m_divisor;
    }

    return static_cast<Bounds<1>>(Range {b0, b0 + bn});
}

static void write_mapping(std::ostream& f, Axis v, int64_t scale, int64_t offset, int64_t divisor) {
    static constexpr const char* variables[] = {"x", "y", "z", "w"};
    const char* var = v.get() < 4 ? variables[v.get()] : "?";

    if (scale != 1) {
        if (offset != 0) {
            f << "(" << scale << "*" << var << " + " << offset << ")";
        } else {
            f << scale << "*" << var;
        }
    } else {
        if (offset != 0) {
            f << "(" << var << " + " << offset << ")";
        } else {
            f << var;
        }
    }

    if (divisor != 1) {
        f << "/" << divisor;
    }
}

std::ostream& operator<<(std::ostream& f, const IndexMap& that) {
    write_mapping(f, that.m_axis, that.m_scale, that.m_offset, that.m_divisor);

    if (that.m_length != 1) {
        f << "...";
        write_mapping(f, that.m_axis, that.m_scale, that.m_offset + that.m_length, that.m_divisor);
    }

    return f;
}

}  // namespace kmm