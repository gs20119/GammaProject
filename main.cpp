
////// include gamma.h ////////

#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include <functional>
#include <stack>
#include <queue>
#include <cmath>
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
        bool original = true;
        int shapeSize(const vector<int>& shape_){
            int s=1; for(int d : shape_) s *= d;
            return s;
        }

    public:
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
            for(int i=0; i<size; i++) storage.get()[i] = x;
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

        ~Tensor(){ storage.reset(); }

        Tensor& operator=(const Tensor& M){
            if(original){ // share storage
                size = M.size; dim = M.dim; offset = M.offset;
                shape = M.shape; stride = M.stride;
                storage.reset(); storage = M.storage; return (*this);
            }
            copy(M);
            return (*this);
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

        void copy(const Tensor& M){
            if(shape != M.shape) cout << "ASSIGNING WRONG SHAPE TO SUBTENSOR" << endl;
            foreach([&M](Tensor& this_, const vector<int>& loc){
                this_(loc) = M(loc);
            });
        }

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
            ref.original = false;
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
        Tensor get(Args... args) const{ return get({args...}); }
        Tensor operator[](Range r) const{ return get({r}); }

        Tensor& operator +=(const Tensor& M){
            if(shape != M.shape) cout << "DIMENSION ERROR WHILE PERFORMING +=" << endl;
            foreach([&M](Tensor& this_, const vector<int>& loc){
                this_(loc) += M(loc);
            });
            return (*this);
        }

        Tensor operator+(const Tensor& M) const{
            if(shape != M.shape) cout << "DIMENSION ERROR WHILE PERFORMING +" << endl;
            return this->copy() += M;
        }

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

        template <typename K> friend class Variable;
        friend class Function;
    };



    class Function;
    class Variadic{
    public:
        typedef shared_ptr<Function> sFunc;
        sFunc creator = nullptr;
        int users, calls = 0;
        bool visit = false;
        Variadic() = default;
        virtual ~Variadic() = default;
    };

    template <typename T>
    class Variable : public Variadic{ // use capsule
    public:
        Tensor<T> body;
        Tensor<T> grad;
        explicit Variable(): body(0.0), grad(0.0,body.shape){}
        explicit Variable(const vector<T>& X) : body(X), grad(0.0,body.shape){}
        explicit Variable(T x, const vector<int>& shape_={1}) : body(x,shape_), grad(0.0,body.shape){}
        Variable(const vector<T>& X, const vector<int>& shape_) : body(X,shape_), grad(0.0,body.shape){}
        explicit Variable(const Tensor<T>& M) : body(M), grad(0.0,body.shape){}

        void backward();
        void print(){ cout << "Variable of " << body; }
        void printGrad(){ cout << "Gradient " << grad; }
        ~Variable(){ cout << "RESET" << endl; }
        friend class Function;
    };

    typedef shared_ptr<Variadic> sVar;
    typedef weak_ptr<Variadic> wVar;
    typedef vector<sVar> Variables;
    template <typename T>
    using something = shared_ptr<Variable<T>>;

    template <typename T, typename... Args>
    something<T> make(Args... args){
        return make_shared<Variable<T>>(args...);
    }

    class Function{
    protected:
        Variables input;
        wVar output;
    public:
        Function() = default;
        virtual ~Function() = default;
        virtual sVar forward(Variables input_){}
        virtual void backward(){}
        template <typename T> friend class Variable;
    };


    template <typename T>
    void Variable<T>::backward(){
        grad = Tensor<T>(1,body.shape);
        queue<sFunc> path;
        path.push(creator); visit=true;
        while(!path.empty()){ // simple BFS
            shared_ptr<Function> f = path.front();
            path.pop();
            for(sVar x : f->input){
                if(!(x->visit)){ x->calls=0; x->users=0; }
                x->users++;
                shared_ptr<Function> g = x->creator;
                if(g != nullptr && !(x->visit)) path.push(g);
                x->visit = true;
            }
        }
        path.push(creator);
        while(!path.empty()){ // topological sort
            shared_ptr<Function> f = path.front();
            path.pop();
            f->backward();
            for(sVar x : f->input){
                x->calls++;
                shared_ptr<Function> g = x->creator;
                if(g != nullptr && x->calls == x->users)
                    path.push(g);
            }
        }
    }


    template <typename T>
    class Square: public Function{
    public:
        Square():Function(){}
        ~Square() override = default;
        sVar forward(Variables input_) override{
            input = input_;
            something<T> X = dynamic_pointer_cast<Variable<T>>(input[0]);
            something<T> Y = make<T>(forward(X->body));
            output = Y;
            return output.lock();
        }
        void backward() override{
            something<T> X = dynamic_pointer_cast<Variable<T>>(input[0]);
            something<T> Y = dynamic_pointer_cast<Variable<T>>(output.lock());
            X->grad += backward(Y->grad, X->body);
        }
        Tensor<T> forward(const Tensor<T>& x) const{
            Tensor<T> y = x.copy();
            y.apply([](T& t){ t = t*t; }); // y = x*x
            return y;
        }
        Tensor<T> backward(const Tensor<T>& dLdy, const Tensor<T>& x) const{
            Tensor<T> dLdx = dLdy.copy();
            dLdx.foreach([&x](Tensor<T>& this_, const vector<int>& loc){
                this_(loc) *= 2*x(loc); // dLdx = 2*x*dLdy
            });
            return dLdx;
        }

        friend something<T> square(something<T> x);
    };

    template <typename T>
    something<T> square(something<T> x_){
        shared_ptr<Function> f = make_shared<Square<T>>();
        Variables X{x_};
        sVar Y = f->forward(X);
        Y->creator = f;
        return dynamic_pointer_cast<Variable<T>>(Y);
    }


    template <typename T>
    class Exp: public Function{
    public:
        Exp():Function(){}
        ~Exp() override = default;
        sVar forward(Variables input_) override{
            input = input_;
            something<T> X = dynamic_pointer_cast<Variable<T>>(input[0]);
            something<T> Y = make<T>(forward(X->body));
            output = Y;
            return output.lock();
        }
        void backward() override{
            something<T> X = dynamic_pointer_cast<Variable<T>>(input[0]);
            something<T> Y = dynamic_pointer_cast<Variable<T>>(output.lock());
            X->grad += backward(Y->grad, X->body);
        }
        Tensor<T> forward(const Tensor<T>& x) const{
            Tensor<T> y = x.copy();
            y.apply([](T& p){ p = exp(p); }); // y = e^x
            return y;
        }
        Tensor<T> backward(const Tensor<T>& dLdy, const Tensor<T>& x) const{
            Tensor<T> dLdx = dLdy.copy();
            dLdx.foreach([&x](Tensor<T>& this_, const vector<int>& loc){
                this_(loc) *= exp(x(loc));
            });
            return dLdx;
        }
        friend something<T> exp(something<T> x);
    };

    template <typename T>
    something<T> exp(something<T> x_){
        shared_ptr<Function> f = make_shared<Exp<T>>();
        Variables X{x_};
        sVar Y = f->forward(X);
        Y->creator = f;
        return dynamic_pointer_cast<Variable<T>>(Y);
    }



    template <typename T>
    class Add: public Function{
    public:
        Add():Function(){}
        ~Add() override = default;
        sVar forward(Variables input_) override{
            input = input_;
            something<T> X1 = dynamic_pointer_cast<Variable<T>>(input[0]);
            something<T> X2 = dynamic_pointer_cast<Variable<T>>(input[1]);
            something<T> Y = make<T>(forward(X1->body, X2->body));
            output = Y;
            return output.lock();
        }
        void backward() override{
            something<T> X1 = dynamic_pointer_cast<Variable<T>>(input[0]);
            something<T> X2 = dynamic_pointer_cast<Variable<T>>(input[1]);
            something<T> Y = dynamic_pointer_cast<Variable<T>>(output.lock());
            X1->grad += backward(Y->grad);
            X2->grad += backward(Y->grad);
        }
        Tensor<T> forward(const Tensor<T>& x1, const Tensor<T>& x2) const{
            Tensor<T> y = x1.copy();
            y.foreach([&x1,&x2](Tensor<T>& this_, const vector<int>& loc){
                this_(loc) = x1(loc)+x2(loc);
            });
            return y;
        }
        Tensor<T> backward(const Tensor<T>& dLdy) const{
            Tensor<T> dLdx = dLdy.copy();
            return dLdx;
        }
        friend something<T> add(something<T> x1, something<T> x2);
    };

    template <typename T>
    something<T> add(something<T> x1, something<T> x2){
        shared_ptr<Function> f = make_shared<Add<T>>();
        Variables X{x1,x2};
        sVar Y = f->forward(X);
        Y->creator = f;
        return dynamic_pointer_cast<Variable<T>>(Y);
    }

}



////////////////////////////////


int main() {
    {
        auto O = gamma::make<float>(0.5);
        auto A = gamma::square(O);
        auto B = gamma::exp(A);
        auto C = gamma::square(B);
        C->print();

        C->backward();
        C->printGrad();
        B->printGrad();
        A->printGrad();
        O->printGrad();
    }

    auto x = gamma::make<double>(2.0);
    auto a = gamma::square(x);
    auto y = gamma::add(gamma::square(a), gamma::square(a));
    y->backward();
    x->print(); a->print(); y->print();
    x->printGrad();



    return 0;
}
