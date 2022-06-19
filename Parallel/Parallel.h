#include "ThreadManager.h"

#ifndef _PARALLEL_
#define _PARALLEL

#include <atomic>
class Parallelizer
{
    public:


        typedef std::uint_fast64_t Type64;

        Parallelizer(const Type64 &_block_size, const Type64 &_rows, const Type64 &_cols);

        ~Parallelizer();

        void Wait();

        void SetBlockSize(const Type64 &_block_size);

        void SetRowsCols(const Type64 &_rows, const Type64 &_cols);


        template <typename F>
        void ParallelizeIndex(const F &task);


        template <typename F>
        void ParallelizeStartEnd(const F &task);

        template <typename F>
        void ParallelizeByRow(const F &task);

    private:

        void DivideBlocks();
        void DivideRows();

        struct block_ranges
        {
            Type64 start;
            Type64 end;
        };

        std::atomic<Type64> blocks_running {0};

        Type64 m_block_size;

        Type64 m_rows;

        Type64 m_cols;

        Type64 num_linear_blocks;

        Type64 num_row_blocks;

        std::unique_ptr<block_ranges[]> linear_ranges;

        std::unique_ptr<block_ranges[]> row_ranges;

        inline static ThreadManager ThreadPool;
};
typedef std::uint_fast64_t Type64;
#include "Parallel.inl"
#endif
