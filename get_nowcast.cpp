// predict_next.cpp
#include <bits/stdc++.h>
using namespace std;

static const double NaN = numeric_limits<double>::quiet_NaN();

static inline string trim(const string& s){
    size_t a=s.find_first_not_of(" \t\r\n"); if(a==string::npos) return "";
    size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1);
}
vector<string> split_csv(const string& s, char delim=','){
    vector<string> out; out.reserve(16);
    string cur; cur.reserve(s.size());
    bool inq=false;
    for(char c: s){
        if(c=='"'){ inq=!inq; continue; }
        if(!inq && c==delim){ out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}
static inline bool to_double(const string& s, double& x){
    string t=trim(s); if(t.empty()){ x=NaN; return false; }
    try{ x=stod(t); return true; } catch(...){ x=NaN; return false; }
}
static inline bool to_int64(const string& s, long long& x){
    string t=trim(s); if(t.empty()){ x=0; return false; }
    try{ x=stoll(t); return true; } catch(...){ x=0; return false; }
}

struct DFRow {
    string instrument;
    string side;
    long long fecha_nano;
    double ts_sec;
    double vwap;
};
struct TP { double t; double v; };

bool read_df_all(const string& path, vector<DFRow>& rows){
    ifstream fin(path);
    if(!fin) return false;
    string header; if(!getline(fin, header)) return false;
    auto cols = split_csv(header);
    for(auto& c: cols) c=trim(c);
    int i_inst=-1, i_side=-1, i_fn=-1, i_ts=-1, i_vwap=-1;
    for(int i=0;i<(int)cols.size();++i){
        if(cols[i]=="instrument") i_inst=i;
        else if(cols[i]=="side") i_side=i;
        else if(cols[i]=="fecha_nano") i_fn=i;
        else if(cols[i]=="ts_sec") i_ts=i;
        else if(cols[i]=="vwap") i_vwap=i;
    }
    if(i_inst<0||i_side<0||i_fn<0||i_ts<0||i_vwap<0){
        cerr<<"df_all.csv no tiene columnas requeridas.\n";
        return false;
    }
    string line;
    while(getline(fin, line)){
        if(trim(line).empty()) continue;
        auto t = split_csv(line);
        if((int)t.size()<=max({i_inst,i_side,i_fn,i_ts,i_vwap}))
            t.resize(max({i_inst,i_side,i_fn,i_ts,i_vwap})+1);
        DFRow r;
        r.instrument = trim(t[i_inst]);
        r.side       = trim(t[i_side]);
        to_int64(t[i_fn], r.fecha_nano);
        to_double(t[i_ts], r.ts_sec);
        to_double(t[i_vwap], r.vwap);
        rows.push_back(move(r));
    }
    return true;
}

bool solve_linear(vector<vector<double>>& A, vector<double>& b, vector<double>& x){
    int n = (int)A.size();
    x.assign(n,0.0);
    for(int i=0;i<n;i++) A[i].push_back(b[i]);
    for(int col=0; col<n; ++col){
        int piv = col;
        double best = fabs(A[col][col]);
        for(int r=col+1; r<n; ++r){
            double v=fabs(A[r][col]);
            if(v>best){ best=v; piv=r; }
        }
        if(best<1e-14) return false;
        if(piv!=col) swap(A[piv], A[col]);
        double div = A[col][col];
        for(int j=col;j<=n;j++) A[col][j] /= div;
        for(int r=0;r<n;r++){
            if(r==col) continue;
            double factor = A[r][col];
            if(fabs(factor)<1e-15) continue;
            for(int j=col;j<=n;j++) A[r][j] -= factor*A[col][j];
        }
    }
    for(int i=0;i<n;i++) x[i]=A[i][n];
    return true;
}

bool fit_line_lastk_at_t(const vector<TP>& s, double t, int k_last, double& y_t, double& slope){
    if((int)s.size()<k_last) return false;
    auto it = upper_bound(s.begin(), s.end(), t, [](double val, const TP& P){ return val < P.t; });
    int end = int(it - s.begin()) - 1;
    if(end < 0) return false;
    int start = end - (k_last - 1);
    if(start < 0) return false;
    double xm = 0.0; for(int i=start;i<=end;i++) xm += s[i].t; xm /= k_last;
    double S00=0, S01=0, S11=0, b0=0, b1=0;
    for(int i=start;i<=end;i++){
        double xc = s[i].t - xm;
        double yi = s[i].v;
        S00 += 1.0;
        S01 += xc;
        S11 += xc*xc;
        b0  += yi;
        b1  += xc*yi;
    }
    double det = S00*S11 - S01*S01;
    if(fabs(det) < 1e-18) return false;
    double a_c = ( b0*S11 - S01*b1) / det;
    double b   = (-b0*S01 + S00*b1) / det;
    y_t  = a_c + b*(t - xm);
    slope = b;
    return true;
}

double tail_median(const vector<double>& d, int W){
    if(d.empty()) return 1.0;
    int n = (int)d.size();
    int m = max(1, min(W, n));
    vector<double> w(d.end()-m, d.end());
    nth_element(w.begin(), w.begin()+m/2, w.end());
    if(m%2==1) return w[m/2];
    nth_element(w.begin(), w.begin()+m/2-1, w.end());
    return 0.5*(w[m/2] + w[m/2-1]);
}

int main(int argc, char** argv){
    string df_path = "df_all.csv";
    string target;
    int k_last = 3;
    int top_others = 4;
    int dt_median_window = 20;
    string xy_out = "xy_train.csv";

    for(int i=1;i<argc;i++){
        string a = argv[i];
        auto need=[&](const char* name){ if(i+1>=argc){ cerr<<"Falta valor para "<<name<<"\n"; exit(1);} return string(argv[++i]); };
        if(a=="--df") df_path = need("--df");
        else if(a=="--target") target = need("--target");
        else if(a=="--k_last") k_last = stoi(need("--k_last"));
        else if(a=="--top_others") top_others = stoi(need("--top_others"));
        else if(a=="--dt_median_window") dt_median_window = stoi(need("--dt_median_window"));
        else if(a=="--xy_out") xy_out = need("--xy_out");
        else { cerr<<"Arg desconocido: "<<a<<"\n"; return 1; }
    }
    if(target.empty()){
        cerr<<"Debes pasar --target <instrumento>\n";
        return 1;
    }

    vector<DFRow> rows;
    if(!read_df_all(df_path, rows)){
        cerr<<"No pude leer "<<df_path<<"\n";
        return 1;
    }

    unordered_map<string, vector<TP>> trade_map;
    for(const auto& r: rows){
        if(r.side!="TRADE") continue;
        if(!isfinite(r.vwap)) continue;
        trade_map[r.instrument].push_back({r.ts_sec, r.vwap});
    }
    for(auto& kv: trade_map){
        auto& v = kv.second;
        sort(v.begin(), v.end(), [](const TP& a, const TP& b){ return a.t < b.t; });
        vector<TP> u; u.reserve(v.size());
        for(const auto& p: v){
            if(!u.empty() && fabs(u.back().t - p.t) < 1e-9) u.back() = p;
            else u.push_back(p);
        }
        v.swap(u);
    }

    if(!trade_map.count(target)){
        cerr<<"Target instrument not found in df_all: "<<target<<"\n";
        return 1;
    }

    vector<pair<string,int>> counts;
    counts.reserve(trade_map.size());
    for(auto& kv: trade_map) counts.push_back({kv.first, (int)kv.second.size()});
    sort(counts.begin(), counts.end(), [](auto& a, auto& b){ return a.second>b.second; });

    vector<string> selected; selected.push_back(target);
    for(auto& pr: counts){
        if((int)selected.size()>=1+top_others) break;
        if(pr.first==target) continue;
        selected.push_back(pr.first);
    }

    unordered_map<string, vector<TP>> dfs_trade;
    for(const auto& inst: selected){
        auto it = trade_map.find(inst);
        if(it!=trade_map.end()) dfs_trade[inst] = it->second;
    }

    const auto& tar = dfs_trade[target];
    if((int)tar.size() < max(k_last, 2)){
        cerr<<"Muy pocos puntos del target.\n";
        return 1;
    }

    struct Row {
        double t0, t1;
        vector<double> p;
        vector<double> m;
        double m_next, dt_next, p_now;
    };
    vector<Row> valid_rows; valid_rows.reserve(tar.size());

    for(int i=k_last-1; i<(int)tar.size()-1; ++i){
        double t0 = tar[i].t;
        double t1 = tar[i+1].t;
        Row row; row.t0=t0; row.t1=t1; row.p.resize(selected.size()); row.m.resize(selected.size());
        bool ok=true;
        for(size_t j=0;j<selected.size();++j){
            const string& inst = selected[j];
            const auto& d = dfs_trade[inst];
            if(inst==target){
                double p_now_true = tar[i].v;
                double yhat, m_now;
                if(!fit_line_lastk_at_t(d, t0, k_last, yhat, m_now)){ ok=false; break; }
                row.p[j] = p_now_true;
                row.m[j] = m_now;
            }else{
                double p_hat, m_hat;
                if(!fit_line_lastk_at_t(d, t0, k_last, p_hat, m_hat)){ ok=false; break; }
                row.p[j] = p_hat;
                row.m[j] = m_hat;
            }
        }
        if(!ok) continue;
        double dummy, m_next;
        if(!fit_line_lastk_at_t(tar, t1, k_last, dummy, m_next)) continue;
        row.m_next = m_next;
        row.dt_next = t1 - t0;
        row.p_now = row.p[0];
        valid_rows.push_back(move(row));
    }

    if(valid_rows.empty()){
        cerr<<"No se generaron muestras válidas.\n";
        return 1;
    }

    int K = (int)selected.size();
    int d = 2*K;

    vector<vector<double>> A(d, vector<double>(d, 0.0));
    vector<double> b(d, 0.0);

    auto add_outer = [&](const vector<double>& x, double y){
        for(int i=0;i<d;i++){
            b[i] += x[i]*y;
            for(int j=0;j<d;j++){
                A[i][j] += x[i]*x[j];
            }
        }
    };

    vector<double> xrow(d);

    ofstream fout(xy_out);
    if(!fout){
        cerr<<"No se pudo abrir "<<xy_out<<" para escritura.\n";
        return 1;
    }
    for(int j=0;j<K;j++){
        if(j) fout<<",";
        fout<<"p__"<<selected[j];
    }
    for(int j=0;j<K;j++){
        fout<<","<<"m__"<<selected[j];
    }
    fout<<",y\n";

    for(const auto& r: valid_rows){
        for(int j=0;j<K;j++) xrow[j]   = r.p[j];
        for(int j=0;j<K;j++) xrow[K+j] = r.m[j];
        for(int i=0;i<d;i++){
            if(i) fout<<",";
            fout<<setprecision(12)<<fixed<<xrow[i];
        }
        fout<<","<<setprecision(12)<<fixed<<r.m_next<<"\n";
        add_outer(xrow, r.m_next);
    }
    fout.close();

    vector<double> beta;
    if(!solve_linear(A, b, beta)){
        cerr<<"No se pudo resolver las ecuaciones normales (matriz singular).\n";
        return 1;
    }

    const auto& r_last = valid_rows.back();
    for(int j=0;j<K;j++) xrow[j]   = r_last.p[j];
    for(int j=0;j<K;j++) xrow[K+j] = r_last.m[j];
    double m_hat = 0.0;
    for(int i=0;i<d;i++) m_hat += xrow[i]*beta[i];

    double last_t0 = r_last.t0;
    int idx_t0 = (int)(upper_bound(tar.begin(), tar.end(), last_t0,
                    [](double val, const TP& P){ return val < P.t; }) - tar.begin()) - 1;
    if(idx_t0 < 1) idx_t0 = 1;
    vector<double> dts; dts.reserve(idx_t0);
    for(int i=1;i<=idx_t0;i++) dts.push_back(tar[i].t - tar[i-1].t);
    double dt_hat = dts.empty()? 1.0 : tail_median(dts, dt_median_window);

    double p0 = r_last.p_now;
    double p_next_hat = p0 + m_hat * dt_hat;

    cout.setf(std::ios::fixed); cout<<setprecision(10);
    cout<<"{\n";
    cout<<"  \"selected_instruments\": [";
    for(int i=0;i<K;i++){
        if(i) cout<<", ";
        cout<<"\""<<selected[i]<<"\"";
    }
    cout<<"],\n";
    cout<<"  \"n_samples\": "<<(int)valid_rows.size()<<",\n";
    cout<<"  \"last_t0\": "<<last_t0<<",\n";
    cout<<"  \"p0\": "<<p0<<",\n";
    cout<<"  \"m_hat\": "<<m_hat<<",\n";
    cout<<"  \"dt_hat_sec\": "<<dt_hat<<",\n";
    cout<<"  \"p_next_hat\": "<<p_next_hat<<",\n";
    cout<<"  \"xy_out\": \""<<xy_out<<"\"\n";
    cout<<"}\n";
    return 0;
}
