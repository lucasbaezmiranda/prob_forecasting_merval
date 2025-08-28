// process_market.cpp
#include <bits/stdc++.h>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;

static const double NaN = std::numeric_limits<double>::quiet_NaN();

/* ---------------- CSV utils ---------------- */
vector<string> split_csv(const string& s, char delim=',') {
    vector<string> out; out.reserve(16);
    string cur; cur.reserve(s.size());
    bool in_quotes = false;
    for (char c : s) {
        if (c == '"') { in_quotes = !in_quotes; continue; }
        if (!in_quotes && c == delim) { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}
static inline string trim(const string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
static inline bool to_double(const string& s, double& x) {
    string t = trim(s);
    if (t.empty()) { x = NaN; return false; }
    try { x = stod(t); return true; }
    catch(...) { x = NaN; return false; }
}
static inline bool to_int64(const string& s, long long& x) {
    string t = trim(s);
    if (t.empty()) { x = 0; return false; }
    try { x = stoll(t); return true; }
    catch(...) { x = 0; return false; }
}

/* ---------------- Data structs ---------------- */
struct RawRow {
    long long fecha_nano; // ns epoch
    double price;
    double quantity;
    string side;
};

struct GroupRow {
    long long fecha_nano;
    string side;
    vector<double> prices;
    vector<double> quantities;
};

struct MetricRow {
    string instrument;
    string side;
    long long fecha_nano;
    double ts_sec;   // fecha_nano / 1e9
    double vwap;
    double spread;   // sqrt(var_ponderada)
};

/* ---------------- IO ---------------- */
bool read_csv_minimal(const string& path,
                      vector<RawRow>& out_rows) {
    ifstream fin(path);
    if (!fin) return false;

    string header;
    if (!getline(fin, header)) return false;

    auto cols = split_csv(header);
    for (auto& c : cols) c = trim(c);

    // map columnas -> índice
    int idx_fecha = -1, idx_price = -1, idx_qty = -1, idx_side = -1;
    for (int i=0;i<(int)cols.size();++i) {
        if (cols[i] == "fecha_nano") idx_fecha = i;
        else if (cols[i] == "price") idx_price = i;
        else if (cols[i] == "quantity") idx_qty = i;
        else if (cols[i] == "side") idx_side = i;
    }
    if (idx_fecha<0 || idx_price<0 || idx_qty<0 || idx_side<0) {
        cerr << "Faltan columnas en: " << path << "\n";
        return false;
    }

    string line;
    while (getline(fin, line)) {
        if (trim(line).empty()) continue;
        auto t = split_csv(line);
        if ((int)t.size() <= max({idx_fecha, idx_price, idx_qty, idx_side}))
            t.resize(max({idx_fecha, idx_price, idx_qty, idx_side})+1);

        long long f; double p, q;
        bool okf = to_int64(t[idx_fecha], f);
        bool okp = to_double(t[idx_price], p);
        bool okq = to_double(t[idx_qty], q);
        string s = trim(t[idx_side]);

        // Simula dropna de Python en (price, quantity, side)
        if (!okf || !okp || !okq || !isfinite(p) || !isfinite(q) || s.empty())
            continue;

        out_rows.push_back({f, p, q, s});
    }
    return true;
}

/* ---------------- Core: build_df_for_side ---------------- */
vector<GroupRow> build_df_for_side(const vector<RawRow>& rows, const string& curr_side) {
    // filtra por side
    vector<RawRow> v;
    v.reserve(rows.size());
    for (const auto& r : rows) if (r.side == curr_side) v.push_back(r);
    if (v.empty()) return {};

    // orden estable por fecha_nano
    stable_sort(v.begin(), v.end(), [](const RawRow& a, const RawRow& b){
        return a.fecha_nano < b.fecha_nano;
    });

    // agrupa por fecha_nano
    vector<GroupRow> out;
    size_t i=0, n=v.size();
    while (i<n) {
        long long key = v[i].fecha_nano;
        size_t j=i;
        vector<double> ps, qs;
        while (j<n && v[j].fecha_nano==key) {
            double p=v[j].price, q=v[j].quantity;
            if (isfinite(p) && isfinite(q) && p>0.0 && q>0.0) {
                ps.push_back(p);
                qs.push_back(q);
            }
            ++j;
        }
        out.push_back({key, curr_side, move(ps), move(qs)});
        i=j;
    }
    return out;
}

/* ---------------- VWAP & spread ---------------- */
inline MetricRow make_metric(const string& instrument, const GroupRow& g) {
    double vwap = NaN, spread = NaN;
    const auto& P = g.prices;
    const auto& W = g.quantities;

    if (!P.empty() && P.size()==W.size()) {
        double sumw = 0.0, sumpw = 0.0;
        for (size_t i=0;i<P.size();++i) {
            double w=W[i], p=P[i];
            if (isfinite(w) && isfinite(p) && w>0.0) { sumw += w; sumpw += w*p; }
        }
        if (sumw > 0.0) {
            vwap = sumpw / sumw;
            double varw = 0.0;
            for (size_t i=0;i<P.size();++i) {
                double w=W[i], p=P[i];
                if (isfinite(w) && isfinite(p) && w>0.0) {
                    double d = p - vwap;
                    varw += w * d * d;
                }
            }
            varw /= sumw;
            spread = std::sqrt(varw);
        }
    }
    double ts_sec = static_cast<double>(g.fecha_nano) / 1e9;
    return {instrument, g.side, g.fecha_nano, ts_sec, vwap, spread};
}

/* ---------------- Main ---------------- */
int main(int argc, char** argv) {
    string dir = "./market_data";
    if (argc>1) {
        string a = argv[1];
        if (a=="--dir" && argc>=3) dir = argv[2];
    }

    // 1) Listar CSVs
    vector<string> csv_files;
    try {
        for (const auto& e : fs::directory_iterator(dir)) {
            if (e.is_regular_file()) {
                auto p = e.path();
                if (p.extension()==".csv") csv_files.push_back(p.string());
            }
        }
    } catch (const std::exception& ex) {
        cerr << "Error leyendo el directorio: " << ex.what() << "\n";
        return 1;
    }
    if (csv_files.empty()) {
        cerr << "No hay CSVs en " << dir << "\n";
        return 1;
    }

    // 2) Leer todos los CSVs
    //    dfs: instrumento (nombre de archivo sin .csv) -> vector<RawRow>
    unordered_map<string, vector<RawRow>> dfs;
    for (const auto& path : csv_files) {
        string stem = fs::path(path).stem().string(); // nombre sin .csv
        vector<RawRow> rows;
        if (!read_csv_minimal(path, rows)) {
            cerr << "Saltando (no legible): " << path << "\n";
            continue;
        }
        if (!rows.empty()) dfs.emplace(stem, move(rows));
    }
    if (dfs.empty()) {
        cerr << "No se pudieron leer filas válidas.\n";
        return 1;
    }

    // 3) Para cada instrumento, detectar sides presentes
    vector<MetricRow> df_all; df_all.reserve(1<<20);
    // all_series[instrument][side]["vwap"/"spread"][fecha_nano] = valor
    struct SeriesPair { map<long long,double> vwap, spread; };
    unordered_map<string, unordered_map<string, SeriesPair>> all_series;

    for (auto& kv : dfs) {
        const string& inst = kv.first;
        const auto& rows = kv.second;

        // sides presentes
        unordered_set<string> sides_present;
        for (const auto& r : rows) sides_present.insert(r.side);

        // build_df_for_side -> group rows por timestamp
        vector<GroupRow> all_groups;
        for (const string& s : sides_present) {
            auto g = build_df_for_side(rows, s);
            all_groups.insert(all_groups.end(), make_move_iterator(g.begin()), make_move_iterator(g.end()));
        }
        if (all_groups.empty()) continue;

        // build_metrics
        // ordenamos por fecha para coherencia
        sort(all_groups.begin(), all_groups.end(),
             [](const GroupRow& a, const GroupRow& b){
                 if (a.fecha_nano!=b.fecha_nano) return a.fecha_nano<b.fecha_nano;
                 return a.side < b.side;
             });

        // Por side, acumulamos series
        unordered_map<string, SeriesPair> series_by_side;

        for (const auto& g : all_groups) {
            MetricRow mr = make_metric(inst, g);
            df_all.push_back(mr);

            // solo cargamos series si vwap/spread no son NaN
            auto& sp = series_by_side[g.side];
            if (isfinite(mr.vwap))   sp.vwap[g.fecha_nano]   = mr.vwap;
            if (isfinite(mr.spread)) sp.spread[g.fecha_nano] = mr.spread;
        }
        all_series[inst] = move(series_by_side);
    }

    if (df_all.empty()) {
        cerr << "df_all vacío (no hubo métricas).\n";
        return 1;
    }

    // 4) Escribir df_all.csv
    {
        ofstream fout("df_all.csv");
        fout << "instrument,side,fecha_nano,ts_sec,vwap,spread\n";
        fout.setf(std::ios::fixed); fout<<setprecision(10);
        for (const auto& r : df_all) {
            fout << r.instrument << "," << r.side << ","
                 << r.fecha_nano << "," << r.ts_sec << ",";
            if (isfinite(r.vwap)) fout << r.vwap; else fout << "";
            fout << ",";
            if (isfinite(r.spread)) fout << r.spread; else fout << "";
            fout << "\n";
        }
    }

    // 5) Elegibles: requieren BI, OF, TRADE con series no vacías
    const unordered_set<string> required_sides = {"BI","OF","TRADE"};
    vector<string> eligible;
    eligible.reserve(all_series.size());
    for (const auto& kv : all_series) {
        const string& inst = kv.first;
        const auto& sides_map = kv.second;

        bool has_all = true;
        for (const auto& rs : required_sides) {
            auto it = sides_map.find(rs);
            if (it==sides_map.end()) { has_all=false; break; }
            const auto& sp = it->second;
            // serie no vacía: al menos un vwap válido
            if (sp.vwap.empty()) { has_all=false; break; }
        }
        if (has_all) eligible.push_back(inst);
    }
    sort(eligible.begin(), eligible.end());

    // 6) Mostrar resumen
    cerr << "DF global escrito en: df_all.csv (rows=" << df_all.size() << ")\n";
    cout << "selected_instruments:\n";
    for (const auto& s : eligible) cout << s << "\n";

    return 0;
}
