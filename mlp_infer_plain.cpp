#include <bits/stdc++.h>
using namespace std;

struct Layer {
    int in_f=0, out_f=0;
    vector<double> W; // row-major: size = in_f*out_f
    vector<double> b; // size = out_f
};
struct Bundle {
    string activation;              // "relu"|"tanh"|"logistic"
    int n_features=0;
    vector<double> scaler_mean;     // n_features
    vector<double> scaler_scale;    // n_features
    vector<Layer> layers;
};

/* ------------------ util csv ------------------ */
static inline string trim(const string& s){
    size_t a=s.find_first_not_of(" \t\r\n"); if(a==string::npos) return "";
    size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1);
}
static vector<string> split(const string& s, char delim){
    vector<string> out; out.reserve(16);
    string cur; cur.reserve(s.size());
    for(char c: s){
        if(c==delim){ out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}
static vector<double> parse_floats_csv_line(const string& line){
    vector<string> t = split(trim(line), ',');
    vector<double> v; v.reserve(t.size());
    for(auto& s : t){
        s = trim(s);
        if(s.empty()){ v.push_back(0.0); continue; }
        v.push_back(strtod(s.c_str(), nullptr));
    }
    return v;
}

/* -------------- activaciones ----------------- */
static inline void relu_inplace(vector<double>& v){
    for(double& x: v) if(x<0.0) x=0.0;
}
static inline void logistic_inplace(vector<double>& v){
    for(double& x: v) x = 1.0 / (1.0 + exp(-x));
}

/* -------------- linalg básica ---------------- */
static vector<double> matmul(const vector<double>& A, int n, int in,
                             const vector<double>& W, int out){
    vector<double> Z((size_t)n*out, 0.0);
    for(int r=0; r<n; ++r){
        const double* arow = &A[(size_t)r*in];
        for(int c=0; c<out; ++c){
            double acc=0.0;
            for(int k=0; k<in; ++k){
                acc += arow[k] * W[(size_t)k*out + c];
            }
            Z[(size_t)r*out + c] = acc;
        }
    }
    return Z;
}
static inline void add_bias_inplace(vector<double>& Z, int n, int out, const vector<double>& b){
    for(int r=0;r<n;++r){
        double* row = &Z[(size_t)r*out];
        for(int c=0;c<out;++c) row[c] += b[c];
    }
}

/* -------------- loader del bundle txt -------- */
static Bundle load_bundle_txt(const string& path){
    ifstream fin(path);
    if(!fin) throw runtime_error("No se pudo abrir: " + path);

    Bundle B;
    string line;

    auto expect_prefix=[&](const string& pref){
        if(!getline(fin, line)) throw runtime_error("Formato invalido (EOF): se esperaba "+pref);
        line = trim(line);
        if(line.rfind(pref,0)!=0) throw runtime_error("Se esperaba prefijo "+pref+", got: "+line);
        return line.substr(pref.size());
    };

    // Cabecera
    B.activation = trim(expect_prefix("ACTIVATION:"));
    B.n_features = stoi(trim(expect_prefix("N_FEATURES:")));

    {
        string rest = trim(expect_prefix("SCALER_MEAN:"));
        B.scaler_mean = parse_floats_csv_line(rest);
        if((int)B.scaler_mean.size()!=B.n_features) throw runtime_error("SCALER_MEAN size != N_FEATURES");
    }
    {
        string rest = trim(expect_prefix("SCALER_SCALE:"));
        B.scaler_scale = parse_floats_csv_line(rest);
        if((int)B.scaler_scale.size()!=B.n_features) throw runtime_error("SCALER_SCALE size != N_FEATURES");
    }
    int L = stoi(trim(expect_prefix("LAYERS:")));
    if(L<=0) throw runtime_error("LAYERS debe ser > 0");

    for(int li=0; li<L; ++li){
        string io = trim(expect_prefix("IN_OUT:"));
        auto io_parts = split(io, ',');
        if(io_parts.size()!=2) throw runtime_error("IN_OUT malformado");
        Layer Lr;
        Lr.in_f  = stoi(trim(io_parts[0]));
        Lr.out_f = stoi(trim(io_parts[1]));
        if(Lr.in_f<=0 || Lr.out_f<=0) throw runtime_error("IN_OUT dims invalidas");
        Lr.W.assign((size_t)Lr.in_f * Lr.out_f, 0.0);

        for(int i=0;i<Lr.in_f;++i){
            string wr = trim(expect_prefix("W_ROW:"));
            auto row = parse_floats_csv_line(wr);
            if((int)row.size()!=Lr.out_f) throw runtime_error("W_ROW size != out_f");
            for(int j=0;j<Lr.out_f;++j) Lr.W[(size_t)i*Lr.out_f + j] = row[j];
        }
        string bline = trim(expect_prefix("B:"));
        Lr.b = parse_floats_csv_line(bline);
        if((int)Lr.b.size()!=Lr.out_f) throw runtime_error("B size != out_f");

        B.layers.push_back(move(Lr));
    }
    if(B.layers.front().in_f != B.n_features)
        throw runtime_error("n_features del scaler != in_f de la primera capa");
    return B;
}

/* -------------- forward ---------------------- */
// X_raw: n x n_features (sin escalar)
static vector<double> mlp_predict(const Bundle& B, const vector<double>& X_raw, int n){
    const int nf = B.n_features;
    if((int)X_raw.size()!=n*nf) throw runtime_error("X_raw size invalido");

    // StandardScaler
    vector<double> A((size_t)n*nf);
    for(int r=0;r<n;++r){
        for(int c=0;c<nf;++c){
            double x = X_raw[(size_t)r*nf + c];
            A[(size_t)r*nf + c] = (x - B.scaler_mean[c]) / B.scaler_scale[c];
        }
    }

    // Capas
    for(size_t li=0; li<B.layers.size(); ++li){
        const Layer& L = B.layers[li];
        vector<double> Z = matmul(A, n, L.in_f, L.W, L.out_f);
        add_bias_inplace(Z, n, L.out_f, L.b);
        const bool last = (li+1==B.layers.size());
        if(!last){
            if(B.activation=="relu") relu_inplace(Z);
            else if(B.activation=="tanh"){ for(double& v: Z) v = tanh(v); }
            else if(B.activation=="logistic") logistic_inplace(Z);
            else throw runtime_error("Activacion no soportada: "+B.activation);
        }
        A.swap(Z);
    }

    // salida lineal; esperamos out_f=1
    int out_f = B.layers.back().out_f;
    vector<double> yhat(n);
    for(int r=0;r<n;++r) yhat[r] = A[(size_t)r*out_f + 0];
    return yhat;
}

/* -------------- leer xy_train.csv (opcional) - header: p__...,m__...,y */
static void read_xy_csv(const string& path, vector<double>& X, vector<double>& y, int& n, int& d){
    ifstream fin(path);
    if(!fin) throw runtime_error("No se pudo abrir: "+path);
    string header; if(!getline(fin, header)) throw runtime_error("CSV vacio");
    auto cols = split(trim(header), ',');
    int y_idx = (int)cols.size()-1;
    if(cols[y_idx]!="y") throw runtime_error("La ultima columna debe llamarse 'y'");

    vector<vector<double>> Xrows;
    vector<double> Y;
    string line;
    while(getline(fin, line)){
        line = trim(line); if(line.empty()) continue;
        auto vals = parse_floats_csv_line(line);
        if((int)vals.size()!= (int)cols.size()) throw runtime_error("Fila con distinto numero de columnas");
        Y.push_back(vals[y_idx]);
        vals.pop_back();
        Xrows.push_back(move(vals));
    }
    n = (int)Xrows.size();
    d = n? (int)Xrows[0].size() : 0;
    X.assign((size_t)n*d, 0.0);
    for(int i=0;i<n;++i){
        if((int)Xrows[i].size()!=d) throw runtime_error("Fila con ancho inconsistente");
        memcpy(&X[(size_t)i*d], Xrows[i].data(), sizeof(double)*d);
    }
    y.swap(Y);
}

/* -------------- métricas opcionales ---------- */
static pair<double,double> mse_r2(const vector<double>& y, const vector<double>& yhat){
    if(y.size()!=yhat.size()) throw runtime_error("mse_r2: tamaño distinto");
    int n = (int)y.size();
    double mse=0.0, mu=0.0;
    for(double v: y) mu+=v; mu/=n;
    double sst=0.0;
    for(int i=0;i<n;++i){
        double e = y[i]-yhat[i];
        mse += e*e;
        double d = y[i]-mu;
        sst += d*d;
    }
    mse/=n;
    double r2 = (sst>0)? 1.0 - n*mse/sst : 0.0;
    return {mse, r2};
}

/* -------------- main ------------------------- */
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if(argc<2){
        cerr<<"Uso:\n"
            <<"  # solo inferencia (lee X de un CSV sin y)\n"
            <<"  ./mlp_infer_plain <mlp_bundle.txt> <X_only.csv>\n\n"
            <<"  # evaluar con xy_train.csv (ultima col y)\n"
            <<"  ./mlp_infer_plain <mlp_bundle.txt> --eval xy_train.csv\n";
        return 1;
    }
    string bundle_path = argv[1];
    Bundle B = load_bundle_txt(bundle_path);

    if(argc>=4 && string(argv[2])=="--eval"){
        string xy_path = argv[3];
        vector<double> X, y; int n=0, d=0;
        read_xy_csv(xy_path, X, y, n, d);
        if(d != B.n_features){
            cerr<<"[ERROR] d="<<d<<" != n_features del modelo "<<B.n_features<<"\n";
            return 1;
        }

        // medir tiempo de inferencia
        auto t0 = chrono::high_resolution_clock::now();
        auto yhat = mlp_predict(B, X, n);
        auto t1 = chrono::high_resolution_clock::now();

        // calcular métricas
        auto [mse, r2] = mse_r2(y, yhat);
        cout.setf(std::ios::fixed); cout<<setprecision(10);
        

        // tiempos
        double elapsed_ms = chrono::duration<double, milli>(t1 - t0).count();
        double per_row_us = (n > 0) ? (elapsed_ms * 1000.0 / n) : 0.0;  // microsegundos por fila

        

        // cantidad a imprimir (por defecto 5)
        int n_print = 5;
        if(argc >= 5) n_print = stoi(argv[4]);

        for(int i=0;i<min(n,n_print);++i){
            cout<<"i="<<i<<"  y="<<y[i]<<"  yhat="<<yhat[i]<<"\n";
        }
        cout<<"\nMSE="<<mse<<"  R2="<<r2<<"\n";
        cout<<"Tiempo total de inferencia: "<<elapsed_ms<<" ms\n";
        cout<<"Tiempo promedio por fila: "<<per_row_us<<" us\n";
        return 0;
    }

    else if(argc>=3){
        // caso: X_only.csv (sin y), con header
        string x_path = argv[2];
        ifstream fin(x_path);
        if(!fin){ cerr<<"No se pudo abrir "<<x_path<<"\n"; return 1; }
        string header; if(!getline(fin, header)){ cerr<<"CSV vacio\n"; return 1; }
        auto cols = split(trim(header), ',');
        int d = (int)cols.size();
        if(d != B.n_features){
            cerr<<"[ERROR] columnas de X="<<d<<" != n_features del modelo "<<B.n_features<<"\n";
            return 1;
        }
        vector<double> X; X.reserve(1<<20);
        string line;
        while(getline(fin, line)){
            line = trim(line); if(line.empty()) continue;
            auto v = parse_floats_csv_line(line);
            if((int)v.size()!=d){ cerr<<"Fila con columnas != d\n"; return 1; }
            X.insert(X.end(), v.begin(), v.end());
        }
        int n = (int)X.size()/d;
        auto yhat = mlp_predict(B, X, n);
        cout.setf(std::ios::fixed); cout<<setprecision(10);
        for(double v: yhat) cout<<v<<"\n";
        return 0;
    }else{
        cerr<<"Argumentos insuficientes.\n";
        return 1;
    }
}
