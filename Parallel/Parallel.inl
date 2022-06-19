
template <typename F>
void Parallelizer::ParallelizeIndex(const F &task)
{
    for (Type64 i = 0; i < num_linear_blocks; i++)
    {
        blocks_running++;

        ThreadPool.Schedule([&start = linear_ranges[i].start, &end =
         linear_ranges[i].end, &task, this]
        {
            for (Type64 j = start; j <= end; j++)
                task(j);
            blocks_running--;
        });
    }
            Wait();
}


template <typename F>
void Parallelizer::ParallelizeStartEnd(const F &task)
{
    for (Type64 i = 0; i < num_linear_blocks; i++)
    {
        blocks_running++;
        ThreadPool.Schedule([&start = linear_ranges[i].start, &end =
        linear_ranges[i].end, &task, this]
        {
                    task(start, end);
                    blocks_running--;
        });
    }
            Wait();
}



template <typename F>
void Parallelizer::ParallelizeByRow(const F &task)
{
    for (Type64 i = 0; i < num_row_blocks; i++)
    {
        blocks_running++;
        ThreadPool.Schedule([&start = row_ranges[i].start, &end =
        row_ranges[i].end, &task, this]
        {
                    task(start, end);
                    blocks_running--;
        });
    }
            Wait();
}
