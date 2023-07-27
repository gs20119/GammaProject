
////// include gamma.h ////////

#include <iostream>
#include <vector>
#include <memory>
using namespace std;

struct Range{
    int from, to;
    Range() : from(0), to(-1) {}
    Range(int x) : from(x), to(x) {}
    Range(vector<int> xy) : from(*(xy.begin())), to(*(xy.end()-1)) {
        if (xy.size() != 2) cout << "ERROR OF RANGE DEFINITION" << endl;
    }
};
Range _;

namespace gamma{

    template <typename T>
    class Tensor{
    private:
        int shapeSize(const vector<int>& shape_){
            int s=1; for(int d : shape_) s *= d;
            return s;
        }

    protected:
        int dim;
        vector<int> shape;
        vector<int> stride;
        int offset, size;
        shared_ptr<T> storage;
        bool origin = true;

    public:

        /*************************************************************************************
         ************************************ Constructor ************************************
         ************************************************************************************/

        explicit Tensor(const vector<T>& X) :
            size(X.size()), dim(1), offset(0), shape({(int)X.size()}),
            stride({1}), storage(new T[size], [](T* a){ delete[] a; }){
            for(int i=0; i<size; i++) storage.get()[i] = X[i];
        }

        Tensor(int x, const vector<int>& shape_) :
            dim(shape_.size()), offset(0), shape(shape_), size(shapeSize(shape)),
            stride(dim), storage(new T[size], [](T* a){ delete[] a; }){
            stride[dim-1] = 1;
            for(int i=dim-2; i>=0; i--) stride[i] = stride[i+1]*shape[i+1];
            if(x!=0) for(int i=0; i<size; i++) storage.get()[i] = x;
        }

        Tensor(const vector<T>& X, const vector<int>& shape_) :
            dim(shape_.size()), offset(0), shape(shape_), size(shapeSize(shape)),
            stride(dim), storage(new T[size], [](T* a){ delete[] a; }){
            if(X.size() != size) cout << "FAILURE TO INITIALIZE GAMMA TENSOR";
            stride[dim-1] = 1;
            for(int i=dim-2; i>=0; i--) stride[i] = stride[i+1]*shape[i+1];
            for(int i=0; i<size; i++) storage.get()[i] = X[i];
        }

        Tensor(const Tensor& M) : // share storage
            size(M.size), dim(M.dim), offset(M.offset), shape(M.shape),
            stride(M.stride), storage(M.storage){}



        /*************************************************************************************
         ***************************** Copy and Assign Operator ******************************
         ************************************************************************************/

        void operator=(const Tensor& M){
            if(origin){ // share storage
                size = M.size; dim = M.dim; offset = M.offset;
                shape = M.shape; stride = M.stride;
                storage.reset(); storage = M.storage; return;
            }
            if(shape != M.shape)
                cout << "ASSIGNING WRONG SHAPE TO SUBTENSOR" << endl;
            vector<int> loc(dim);
            for(int i=0; i<size; i++){
                (*this)(loc) = M(loc); loc[dim-1]++;
                for(int j=dim-1; j>0; j--)
                    if(loc[j]==shape[j]){ loc[j]=0; loc[j-1]++; }
            }
        }

        Tensor copy() const{
            Tensor M(0,shape);
            vector<int> loc(dim);
            vector<int> iStride(dim); iStride[dim-1]=1;
            for(int i=dim-2; i>=0; i--) iStride[i]=iStride[i+1]*shape[i+1];
            for(int i=0; i<M.size; i++){
                M.storage.get()[i] = (*this)(loc); loc[dim-1]++;
                for(int j=dim-1; j>0; j--)
                    if(loc[j]==shape[j]){ loc[j]=0; loc[j-1]++; }
            }
            return M;
        }



        /*************************************************************************************
         ******************************* Indexing and Slicing ********************************
         ************************************************************************************/

        T& operator()(const vector<int>& loc) const{
            int idx = offset;
            for(int i=0; i<dim; i++){
                if(shape[i] <= loc[i]) cout << "INDEX OUT OF RANGE" << endl;
                idx += stride[i]*loc[i];
            }
            return storage.get()[idx];
        }

        template <typename... Args>
        T& operator()(Args... args) const{ // access by index
            vector<int> loc{ args... };
            return (*this)(loc);
        }

        Tensor get(const vector<Range>& index) const{ // slicing, origin = false
            Tensor ref = Tensor(*this);
            ref.origin = false;
            if(index.size() > dim) cout << "INDEX DIMENSION ERROR" << endl;
            ref.dim = 0;
            ref.stride.clear(); ref.shape.clear();
            for(int i=0; i<dim; i++){
                int width = shape[i];
                int from = (index.size() <= i) ? 0 : ((index[i].from%width)+width)%width;
                int to = (index.size() <= i) ? width-1 : ((index[i].to%width)+width)%width;
                ref.offset += from*stride[i];
                if(from > to) cout << "RANGE DOESN'T CONTAIN ANYTHING" << endl;
                if(from < to){
                    ref.shape.push_back(to-from+1); ref.dim++;
                    ref.stride.push_back(stride[i]);
                }
            }ref.size = 1;
            for(int d : ref.shape) ref.size *= d;
            return ref;
        }

        template <typename... Args>
        Tensor get(Args... args){
            vector<Range> index{args...};
            return get(index);
        }

        Tensor operator[](Range r) const{
            vector<Range> order{r};
            return get(order);
        }



        /*************************************************************************************
         *************************************** Misc. ***************************************
         ************************************************************************************/

        friend ostream& operator<<(ostream& out, Tensor M){
            cout << "[";
            vector<int> loc(M.dim);
            for(int i=0; i<M.size; i++){
                for(int j=0; j<M.dim-1; j++) if(i % M.stride[j] == 0) cout << "[";
                cout << M(loc);
                for(int j=0; j<M.dim-1; j++) if((i+1) % M.stride[j] == 0) cout << "]";
                if(i != M.size-1) cout << ", ";
                for(int j=M.dim-1; j>0; j--)
                    if(loc[j]==M.shape[j]){ loc[j]=0; loc[j-1]++; }
            }
            cout << "]" << endl;
            return out;
        }

        template <typename T_> friend class Variable;
        template <typename From, typename To> friend class Function;

        ~Tensor(){ storage.reset(); }
    };



    template <typename T>
    class Variable{
    private:
        Tensor<T> body;
        Tensor<T> grad;
    public:

    };

    template <typename From, typename To>
    class Function{

    public:
        To operator()(const From& x){};
    };

}



////////////////////////////////


int main() {
    clock_t start, fin;
    double duration;
    start = clock();

    for(int ITER=0; ITER<30; ITER++){
        gamma::Tensor<int> T(0,{10, 1000, 1000});
        for(int i=0; i<10; i++){
            gamma::Tensor<int> X(i,{1000, 1000});
            for(int j=0; j<1000; j++) X(j,j) = 0;
            T[i] = X.copy();
        }
    }

    fin = clock();
    duration = (double)(fin-start)/CLOCKS_PER_SEC;
    cout << duration << endl;
    return 0;
}
