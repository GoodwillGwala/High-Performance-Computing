#include "matrix.h"

template <typename T>
matrix<T>::matrix()
: m_rows(0), m_cols(0), mat(new T[0])
, parallelizer(global_block_size, 0, 0)
{
    elements = mat.get();
}


template <typename T>
matrix<T>::matrix(const Type64 &_rows, const Type64 &_cols)
: m_rows(_rows), m_cols(_cols), mat(new T[_rows * _cols])
, parallelizer(global_block_size, _rows, _cols)
{
    if (m_rows == 0 or m_cols == 0)
       throw zero_size{};
    elements = mat.get();
}

template <typename T>
matrix<T>::matrix(const Type64 &_rows, const Type64 &_cols, const T &scalar)
: m_rows(_rows), m_cols(_cols), mat(new T[_rows * _cols])
, parallelizer(global_block_size, _rows, _cols)
{
    if (m_rows == 0 or m_cols == 0)
       throw zero_size{};
    elements = mat.get();
    Fill(scalar);
}

template <typename T>
matrix<T> &matrix<T>::operator=(const matrix<T> &m)
{
    m_rows = m.m_rows;
    m_cols = m.m_cols;
    mat.reset(new T[m_rows * m_cols]);
    elements = mat.get();
    parallelizer.SetRowsCols(m_rows, m_cols);
    Copy(m.elements, elements);
    return *this;
}


template <typename T>
matrix<T>::matrix(const matrix<T> &m)
: m_rows(m.m_rows), m_cols(m.m_cols), mat(new T[m.m_rows * m.m_cols])
, parallelizer(global_block_size, m.m_rows, m.m_cols)
{
    elements = mat.get();
    Copy(m.elements, elements);
}

template <typename T>
matrix<T>& matrix<T>::operator=(matrix<T> &&m)
{
    m_rows = m.m_rows;
    m_cols = m.m_cols;
    mat = std::move(m.mat);
    elements = mat.get();
    parallelizer.SetRowsCols(m_rows, m_cols);
    m.parallelizer.SetRowsCols(0, 0);
    m.m_rows = 0;
    m.m_cols = 0;
    m.elements = nullptr;
    return *this;
}


template <typename T>
void matrix<T>::Fill(const T &scalar)
{
    parallelizer.ParallelizeIndex([this, &scalar](const Type64 &i)
    {
        elements[i] = scalar;
    });
}

template <typename T>
Type64 matrix<T>::get_cols() const
{
    return m_cols;
}


template <typename T>
Type64 matrix<T>::get_rows() const
{
    return m_rows;
}

template <typename T>
matrix<T> matrix<T>::Transpose()
{
    matrix<T> out(m_cols, m_rows);
    out.parallelizer.ParallelizeByRow([this, &out](const Type64 &start, const Type64 &end)
    {
        for (Type64 i = start; i <= end; i++)
            for (Type64 j = 0; j < out.m_cols; j++)
                 out(i, j) = operator()(j, i);

    });
    return out;
}



template <typename T>
matrix<T>& matrix<T>::Transpose(matrix<T>& out)
{

    parallelizer.ParallelizeByRow([this, &out](const Type64 &start, const Type64 &end)
    {
        for (Type64 i = start; i <= end; i++)
            for (Type64 j = 0; j < out.m_cols; j++)
                 out(i, j) = operator()(j, i);

    });
    return out;
}



template <typename T>
T &matrix<T>::operator()(const Type64 &row, const Type64 &col)
{
    return elements[(m_cols * row) + col];
}


template <typename T>
T matrix<T>::operator()(const Type64 &row, const Type64 &col) const
{
    return elements[(m_cols * row) + col];
}

template <typename T>
T matrix<T>::operator[](const Type64 &i) const
{
    return elements[i];
}

template <typename T>
T& matrix<T>::at(const Type64 &row, const Type64 &col)
{
    if (row >= m_rows or col >= m_cols)
        throw index_out_of_range{};

    return elements[(m_cols * row) + col];
}

template <typename T>
T matrix<T>::at(const Type64 &row, const Type64 &col) const
{
    if (row >= m_rows or col >= m_cols)
        throw index_out_of_range{};

    return elements[(m_cols * row) + col];
}


template <typename T>
void matrix<T>::Copy(const T *in, T *out)
{
    if (in == out) return;
    parallelizer.ParallelizeIndex([in, out](const Type64 &i)
    {
        out[i] = in[i];
    });
}
