
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
        vector<int> iStride;
        int offset, size;
        shared_ptr<T> storage;
        bool origin = true;

    public:

        /*************************************************************************************
         ************************************ Constructor ************************************
         ************************************************************************************/

        explicit Tensor(const vector<T>& X) :
            size(X.size()), dim(1), offset(0), shape({(int)X.size()}),
            stride({1}), iStride({1}), storage(new T[size], [](T* a){ delete[] a; }){
            for(int i=0; i<size; i++) storage.get()[i] = X[i];
        }

        Tensor(int x, const vector<int>& shape_) :
            dim(shape_.size()), offset(0), shape(shape_), size(shapeSize(shape)),
            stride(dim), storage(new T[size], [](T* a){ delete[] a; }){
            stride[dim-1] = 1;
            for(int i=dim-2; i>=0; i--) stride[i] = stride[i+1]*shape[i+1]; iStride = stride;
            for(int i=0; i<size; i++) storage.get()[i] = x;
        }

        Tensor(const vector<T>& X, const vector<int>& shape_) :
            dim(shape_.size()), offset(0), shape(shape_), size(shapeSize(shape)),
            stride(dim), storage(new T[size], [](T* a){ delete[] a; }){
            if(X.size() != size) cout << "FAILURE TO INITIALIZE GAMMA TENSOR";
            stride[dim-1] = 1;
            for(int i=dim-2; i>=0; i--) stride[i] = stride[i+1]*shape[i+1]; iStride = stride;
            for(int i=0; i<size; i++) storage.get()[i] = X[i];
        }

        Tensor(const Tensor& M) : // share storage
            size(M.size), dim(M.dim), offset(M.offset), shape(M.shape),
            stride(M.stride), iStride(M.iStride), storage(M.storage){}



        /*************************************************************************************
         ***************************** Copy and Assign Operator ******************************
         ************************************************************************************/

        void operator=(const Tensor& M){
            if(origin){ // share storage
                size = M.size; dim = M.dim; offset = M.offset;
                shape = M.shape; stride = M.stride; iStride = M.iStride;
                storage.reset(); storage = M.storage; return;
            }
            if(shape != M.shape)
                cout << "ASSIGNING WRONG SHAPE TO SUBTENSOR" << endl;
            vector<int> loc(dim);
            for(int i=0; i<size; i++){
                loc[0] = (int)i/M.iStride[0];
                for(int j=1; j<dim; j++)
                    loc[j] = (int)(i%M.iStride[j-1])/M.iStride[j];
                (*this)(loc) = M(loc);
            }
        }

        Tensor copy() const{
            Tensor M(0,shape);
            vector<int> loc(dim);
            vector<int> iStride(dim); iStride[dim-1]=1;
            for(int i=dim-2; i>=0; i--) iStride[i]=iStride[i+1]*shape[i+1];
            for(int i=0; i<M.size; i++){
                loc[0] = (int)i/M.iStride[0];
                for(int j=1; j<M.dim; j++)
                    loc[j] = (int)(i%M.iStride[j-1])/M.iStride[j];
                M.storage.get()[i] = (*this)(loc);
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

        Tensor operator[](const vector<Range>& order) const{ // slicing, origin = false
            Tensor ref = Tensor(*this);
            ref.origin = false;
            if(order.size() > dim) cout << "INDEX DIMENSION ERROR" << endl;
            ref.dim = 0;
            ref.stride.clear(); ref.shape.clear();
            for(int i=0; i<dim; i++){
                int width = shape[i];
                int from = (order.size() <= i) ? 0 : ((order[i].from%width)+width)%width;
                int to = (order.size() <= i) ? width-1 : ((order[i].to%width)+width)%width;
                ref.offset += from*stride[i];
                if(from > to) cout << "RANGE DOESN'T CONTAIN ANYTHING" << endl;
                if(from < to){
                    ref.shape.push_back(to-from+1); ref.dim++;
                    ref.stride.push_back(stride[i]);
                }
            }
            ref.iStride = vector<int>(ref.dim); ref.iStride[ref.dim-1]=1;
            for(int i=ref.dim-2; i>=0; i--)
                ref.iStride[i]=ref.iStride[i+1]*ref.shape[i+1];
            ref.size = 1; for(int d : ref.shape) ref.size *= d;
            return ref;
        }

        Tensor operator[](Range r) const{
            vector<Range> order{r};
            return (*this)[order];
        }



        /*************************************************************************************
         *************************************** Misc. ***************************************
         ************************************************************************************/

        friend ostream& operator<<(ostream& out, Tensor M){
            cout << "[";
            vector<int> loc(M.dim);
            for(int i=0; i<M.size; i++){
                loc[0] = (int)i/M.iStride[0];
                for(int j=1; j<M.dim; j++) loc[j] = (int)(i%M.iStride[j-1])/M.iStride[j];
                for(int j=0; j<M.dim-1; j++) if(i % M.stride[j] == 0) cout << "[";
                cout << M(loc);
                for(int j=0; j<M.dim-1; j++) if((i+1) % M.stride[j] == 0) cout << "]";
                if(i != M.size-1) cout << ", ";
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
    vector<int> V(18);
    vector<int> D = {3,2,3};
    gamma::Tensor<int> X(V, D);
    gamma::Tensor<int> Y({1,2,3,4}, {2,2});
    gamma::Tensor<int> Z({1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7}, {3,3,3});
    cout << X << Y << Z << endl;
    Y = Z[0];
    cout << Y;
    Y(1,0) = 10;
    cout << Z;
    Y[{_,2}] = gamma::Tensor<int>({20,20,20});
    cout << Z;
    return 0;
}
