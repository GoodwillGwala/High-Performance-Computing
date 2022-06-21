#ifndef _MATRIX_OP
#define _MATRIX_OP


#include <algorithm>


#include <iomanip>

#include <memory>
#include <mutex>

#include <random>
#include <sstream>

#include <utility>
#include "Parallel.cpp"
//#include "matrix.inl"


//typedef std::uint_fast32_t ui32;


//Class Exceptions
    class index_out_of_range{};
    class not_square{};
    class zero_size{};


template <typename T>
class matrix
{
    public:

//defualt constructor
        matrix();


//parameter constructor
        matrix(const Type64 &_rows, const Type64 &_cols);


//initialized constructor
        matrix(const Type64 &_rows, const Type64 &_cols, const T &scalar);

//Copy constructor
        matrix<T> &operator=(const matrix<T> &m);

//move constructors
        matrix<T> &operator=(matrix<T> &&m);


         matrix(const matrix<T> &m);

        void Fill(const T &scalar);


        Type64 get_cols() const;


        Type64 get_rows() const;


        matrix<T> Transpose();
        matrix<T>& Transpose(matrix<T>& out);



        T &operator()(const Type64 &row, const Type64 &col);

        T operator()(const Type64 &row, const Type64 &col) const;


        T &operator[](const Type64 &i);


        T operator[](const Type64 &i) const;



        T &at(const Type64 &row, const Type64 &col);


        T at(const Type64 &row, const Type64 &col) const;



        inline static Type64 global_block_size = {0};

        //TODO define in .cpp, nested template classes can be a pain
        template <typename D>
        class random_generator
        {
            public:

                template <typename... P>
                random_generator(P... params) : dist(params...) {}

                T generate_scalar()
                {
                    static std::mt19937_64 mt(rd());
                    return dist(mt);
                }


                matrix<T> generate_matrix(const Type64 &rows, const Type64 &cols)
                {
                    matrix<T> m(m_rows, m_cols);
                    randomize_matrix(m);
                    return m;
                }


                void randomize_matrix(matrix<T> &m)
                {

                    m.parallelizer.ParallelizeStartEnd([this, &m](const Type64 &start, const Type64 &end)
                    {
                        std::mt19937_64 mt(generate_seed());
                        for (Type64 i = start; i <= end; i++)
                            m.elements[i] = dist(mt);
                    });
                }

            private:

                Type64 generate_seed()
                {
                    static std::mt19937_64 mt(rd());
                    return mt();
                }

                D dist;

                std::random_device rd;
        };


    private:

        void Copy(const T *in, T *out);

        Type64 m_rows = 0;

        Type64 m_cols = 0;

        T *elements = nullptr;

        std::unique_ptr<T[]> mat;

        Parallelizer parallelizer;

};
typedef std::uint_fast64_t Type64;
#endif
