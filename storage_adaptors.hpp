#ifndef BOOST_UBLAS_STORAGE_ADAPTORS_H
#define BOOST_UBLAS_STORAGE_ADAPTORS_H

#include <algorithm>

#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/detail/iterator.hpp>


namespace boost { namespace numeric { namespace ublas {


    template<class T>
    class readonly_array_adaptor:
        public storage_array<readonly_array_adaptor<T> > {

        typedef readonly_array_adaptor<T> self_type;
    public:
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef T value_type;
        typedef const T &const_reference;
        typedef const T *const_pointer;
    public:

        // Construction and destruction
        BOOST_UBLAS_INLINE
        readonly_array_adaptor ():
            size_ (0), data_ (0) {
        }
        BOOST_UBLAS_INLINE
        readonly_array_adaptor (size_type size, const_pointer data):
            size_ (size), data_ (data) {
        }
        BOOST_UBLAS_INLINE
        ~readonly_array_adaptor () {
        }

        readonly_array_adaptor (const readonly_array_adaptor& rhs)
          : size_(rhs.size_), data_(rhs.data_)
        { }

        // Resizing
        BOOST_UBLAS_INLINE
        void resize (size_type size) {
            size_ = size;
        }
        BOOST_UBLAS_INLINE
        void resize (size_type size, const_pointer data) {
            size_ = size;
            data_ = data;
        }

        // Random Access Container
        BOOST_UBLAS_INLINE
        size_type max_size () const {
            return std::numeric_limits<size_type>::max ();
        }

        BOOST_UBLAS_INLINE
        bool empty () const {
            return size_ == 0;
        }

        BOOST_UBLAS_INLINE
        size_type size () const {
            return size_;
        }

        // Element access
        BOOST_UBLAS_INLINE
        const_reference operator [] (size_type i) const {
            BOOST_UBLAS_CHECK (i < size_, bad_index ());
            return data_ [i];
        }

        // Iterators simply are pointers.
        typedef const_pointer const_iterator;

        BOOST_UBLAS_INLINE
        const_iterator begin () const {
            return data_;
        }
        BOOST_UBLAS_INLINE
        const_iterator end () const {
            return data_ + size_;
        }

        // this typedef is used by vector and matrix classes
        typedef const_pointer iterator;

        // Reverse iterators
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;

        BOOST_UBLAS_INLINE
        const_reverse_iterator rbegin () const {
            return const_reverse_iterator (end ());
        }
        BOOST_UBLAS_INLINE
        const_reverse_iterator rend () const {
            return const_reverse_iterator (begin ());
        }

    private:
        size_type size_;
        const_pointer data_;
    };

    template <class T>
    vector<const T, readonly_array_adaptor<T> >
    make_vector_from_pointer(const size_t size, const T * data)
    {
        typedef readonly_array_adaptor<T> a_t;
        typedef vector<const T, a_t>      v_t;
        return v_t(size, a_t(size, data));
    }

    template <class LAYOUT, class T>
    matrix<const T, LAYOUT, readonly_array_adaptor<T> >
    make_matrix_from_pointer(const size_t size1, const size_t size2, const T * data)
    {
        typedef readonly_array_adaptor<T> a_t;
        typedef matrix<const T, LAYOUT, a_t>      m_t;
        return m_t(size1, size2, a_t(size1*size2, data));
    }
    // default layout: row_major
    template <class T>
    matrix<const T, row_major, readonly_array_adaptor<T> >
    make_matrix_from_pointer(const size_t size1, const size_t size2, const T * data)
    {
	  return make_matrix_from_pointer<row_major>(size1, size2, data);
    }

    template <class T, size_t M, size_t N>
    matrix<const T, row_major, readonly_array_adaptor<T> >
    make_matrix_from_pointer(const T (&array)[M][N])
    {
        typedef readonly_array_adaptor<T> a_t;
        typedef matrix<const T, row_major, a_t>      m_t;
        return m_t(M, N, a_t(M*N, array[0]));
    }
    template <class T, size_t M, size_t N>
    matrix<const T, row_major, readonly_array_adaptor<T> >
    make_matrix_from_pointer(const T (*array)[M][N])
    {
        typedef readonly_array_adaptor<T> a_t;
        typedef matrix<const T, row_major, a_t>      m_t;
        return m_t(M, N, a_t(M*N, (*array)[0]));
    }

}}}

#endif
