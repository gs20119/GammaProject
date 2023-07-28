
////// include gamma.h ////////

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <math.h>
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
        int dim;
        vector<int> shape;
        vector<int> stride;
        int offset, size;
        shared_ptr<T> storage;
        bool origin = true;
        int shapeSize(const vector<int>& shape_){
            int s=1; for(int d : shape_) s *= d;
            return s;
        }

    public:

        /*************************************************************************************
         ************************************ Constructor ************************************
         ************************************************************************************/

        explicit Tensor(const vector<T>& X) :
            size(X.size()), dim(1), offset(0), shape({(int)X.size()}),
            stride({1}), storage(new T[size], [](T* a){ delete[] a; }){
            for(int i=0; i<size; i++) storage.get()[i] = X[i];
        }

        explicit Tensor(T x, const vector<int>& shape_={1}) :
            dim(shape_.size()), offset(0), shape(shape_), size(shapeSize(shape)),
            stride(dim), storage(new T[size], [](T* a){ delete[] a; }){
            stride[dim-1] = 1;
            for(int i=dim-2; i>=0; i--) stride[i] = stride[i+1]*shape[i+1];
            if(x!=0) for(int i=0; i<size; i++) storage.get()[i] = x;
        }

        Tensor(const vector<T>& X, const vector<int>& shape_) :
            dim(shape_.size()), offset(0), shape(shape_), size(shapeSize(shape)),
            stride(dim), storage(new T[size], [](T* a){ delete[] a; }){
            if(X.size() != size) cout << "FAILURE TO INITIALIZE GAMMA TENSOR" << endl;
            stride[dim-1] = 1;
            for(int i=dim-2; i>=0; i--) stride[i] = stride[i+1]*shape[i+1];
            for(int i=0; i<size; i++) storage.get()[i] = X[i];
        }

        Tensor(const Tensor& M) : // share storage
            size(M.size), dim(M.dim), offset(M.offset), shape(M.shape),
            stride(M.stride), storage(M.storage){}



        /*************************************************************************************
         ******************************** Copy and Assignment ********************************
         ************************************************************************************/

        void operator=(const Tensor& M){
            if(origin){ // share storage
                size = M.size; dim = M.dim; offset = M.offset;
                shape = M.shape; stride = M.stride;
                storage.reset(); storage = M.storage; return;
            }
            if(shape != M.shape) cout << "ASSIGNING WRONG SHAPE TO SUBTENSOR" << endl;
            foreach([&M](Tensor& this_, const vector<int>& loc){
                this_(loc) = M(loc);
            });
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
         ********************** Operations that Preserve the Original ************************
         ************************************************************************************/

        Tensor operator+(const Tensor M) const{
            Tensor out = (*this).copy();
            if(shape != M.shape) cout << "DIMENSION ERROR WHILE PERFORMING +" << endl;
            out.foreach([&M](Tensor& this_, const vector<int>& loc){
                this_(loc) += M(loc);
            });
            return out;
        }



        /*************************************************************************************
         ***************************** Operations that Doesn't *******************************
         ************************************************************************************/

        void foreach(function<void(Tensor&, const vector<int>&)> process){
            vector<int> loc(dim);
            for(int i=0; i<size; i++){
                process(*this, loc); loc[dim-1]++;
                for(int j=dim-1; j>0; j--)
                    if(loc[j]==shape[j]){ loc[j]=0; loc[j-1]++; }
            }
        }

        void apply(function<void(T&)> func){
            foreach([&func](Tensor& this_, const vector<int>& loc){
                func(this_(loc));
            });
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
                if(i != M.size-1) cout << ", "; loc[M.dim-1]++;
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
    public:
        Tensor<T> body;
        Tensor<T> grad;

        explicit Variable(): body(0), grad(0,body.shape){}
        explicit Variable(const vector<T>& X) : body(X), grad(0,body.shape){}
        explicit Variable(T x, const vector<int>& shape_={1}) : body(x,shape_), grad(0,body.shape){}
        Variable(const vector<T>& X, const vector<int>& shape_) : body(X,shape_), grad(0,body.shape){}
        explicit Variable(const Tensor<T>& M) : body(M), grad(0,body.shape){}

        Variable(const Variable& x) : body(x.body), grad(x.grad){}
        void operator=(const Variable& x){ body = x.body; grad = x.grad; }

        friend ostream& operator<<(ostream& out, Variable M){
            cout << "Variable of " << M.body;
            return out;
        }
        template <typename From, typename To> friend class Function;

    };


    template <typename From, typename To>
    class Function{
    protected:
        Variable<From> input;
    public:
        Variable<To> operator()(const Variable<From>& input_){
            input = input_;
            Tensor<From> x = input.body;
            Tensor<To> y = forward(x);
            return Variable<To>(y);
        }
        virtual Tensor<To> forward(const Tensor<From> x) const {}
        virtual Tensor<From> backward(const Tensor<To> dLdy) const {}
    };

    template <typename T>
    class Square: public Function<T,T>{
    public:
        Tensor<T> forward(const Tensor<T> x) const override{
            Tensor<T> y = x.copy();
            y.apply([](T& t){ t = t*t; }); // y = x*x
            return y;
        }
        Tensor<T> backward(const Tensor<T> dLdy) const override{
            Tensor<T> dLdx = dLdy.copy();
            Tensor<T> x = Function<T,T>::input.body;
            dLdx.foreach([&x](Tensor<T>& this_, const vector<int>& loc){
                this_(loc) *= 2*x(loc); // dLdx = 2*x*dLdy
            });
            return dLdx;
        }
    };

    template <typename T>
    class Exp: public Function<T,T>{
    public:
        Tensor<T> forward(const Tensor<T> x) const override{
            Tensor<T> y = x.copy();
            y.apply([](T& p){ p = exp(p); }); // y = e^x
            return y;
        }
        Tensor<T> backward(const Tensor<T> dLdy) const override{
            Tensor<T> dLdx = dLdy.copy();
            Tensor<T> x = Function<T,T>::input.body;
            dLdx.foreach([&x](Tensor<T>& this_, const vector<int>& loc){
                this_(loc) *= exp(x(loc));
            });
            return dLdx;
        }
    };

}



////////////////////////////////


int main() {

    gamma::Variable<float> x(0.5);
    cout << x;

    auto f = gamma::Square<float>();
    auto g = gamma::Exp<float>();
    auto h = gamma::Square<float>();

    auto A = f(x);
    auto B = g(A);
    auto C = h(B);
    cout << C;

    C.grad = gamma::Tensor<float>(1);
    B.grad = h.backward(C.grad);
    A.grad = g.backward(B.grad);
    x.grad = f.backward(A.grad);
    cout << x.grad;

    return 0;
}
