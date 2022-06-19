#include "Parallel.h"

Parallelizer::Parallelizer(const Type64& block_size, const Type64& rows, const Type64& cols)
: m_block_size(block_size)
, m_rows(rows)
, m_cols(cols)
{
    DivideBlocks();
    DivideRows();
}

Parallelizer::~Parallelizer()
{
    Wait();
}

void Parallelizer::Wait()
{
    while (blocks_running != 0)
        std::this_thread::yield();

}

void Parallelizer::SetBlockSize(const Type64& block_size)
{
    m_block_size = block_size;
    DivideBlocks();
    DivideRows();
}

void Parallelizer::SetRowsCols(const Type64& rows, const Type64& cols)
{
    m_rows = rows;
    m_cols = cols;
    DivideBlocks();
    DivideRows();
}


void Parallelizer::DivideBlocks()
{
    if (m_rows == 0 or m_cols == 0)
    {
        num_linear_blocks = 0;
        linear_ranges.reset(nullptr);
    }
    else
    {
        Type64 use_block_size = m_block_size ? m_block_size :
        std::max<Type64>(m_rows * m_cols / ThreadPool.ThreadCount(), 1);

        num_linear_blocks = m_rows * m_cols / use_block_size;

        if ((m_rows * m_cols) - (num_linear_blocks * use_block_size) != 0)
            num_linear_blocks++;

        if (num_linear_blocks == 0)
            num_linear_blocks = 1;

        linear_ranges.reset(new block_ranges[num_linear_blocks]);

        for (Type64 i = 0; i < num_linear_blocks; i++)
        {
            linear_ranges[i].start = i * use_block_size;
            if (i == num_linear_blocks - 1)
                linear_ranges[i].end = (m_rows * m_cols) - 1;
            else
                linear_ranges[i].end = ((i + 1) * use_block_size) - 1;
       }
    }
}

void Parallelizer::DivideRows()
{
    if (m_rows == 0 or m_cols == 0)
    {
        num_row_blocks = 0;
        row_ranges.reset(nullptr);
    }
    else
    {
        Type64 use_block_size = m_block_size ? m_block_size :
        std::max<Type64>(m_rows * m_cols / ThreadPool.ThreadCount(), 1);

        Type64 rows_per_block = use_block_size / m_cols;
        if (rows_per_block == 0)
        {
            rows_per_block = 1;
            num_row_blocks = m_rows;
        }
        else
            num_row_blocks = m_rows / rows_per_block;

        if (num_row_blocks == 0)
            num_row_blocks = 1;

        row_ranges.reset(new block_ranges[num_row_blocks]);

        for (Type64 i = 0; i < num_row_blocks; i++)
        {
            row_ranges[i].start = i * rows_per_block;
            if (i == num_row_blocks - 1)
            row_ranges[i].end = m_rows - 1;
            else
            row_ranges[i].end = ((i + 1) * rows_per_block) - 1;
        }
    }
}
