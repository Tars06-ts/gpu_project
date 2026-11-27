#include "common/loader.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& file, long line, const std::string& msg){
	throw std::runtime_error(file + ":" + std::to_string(line) + ": " + msg);
}

inline bool is_sep(char c){ return c == ' ' || c == '\t' || c == ',' || c == '\r'; }
inline bool is_comment(char c){ return c == '#' || c == '%'; }

// splits on spaces, tabs and commas, and stops at a comment character
void tokenize(const std::string& line, std::vector<std::string_view>& tok){
	tok.clear();
	const size_t n = line.size();
	size_t i = 0;
	while (i < n){
		if (is_comment(line[i])) break;
		if (is_sep(line[i])){ i++; continue; }
		size_t j = i;
		while (j < n && !is_sep(line[j]) && !is_comment(line[j])) j++;
		tok.emplace_back(line.data() + i, j - i);
		i = j;
	}
}

bool to_int(std::string_view s, int& out){
	if (s.empty()) return false;
	const char* b = s.data();
	const char* e = b + s.size();
	auto r = std::from_chars(b, e, out);
	return r.ec == std::errc() && r.ptr == e;
}

bool is_number(std::string_view s){
	int t;
	return to_int(s, t);
}

bool is_blank(const std::string& line){
	for (char c : line){
		if (c != ' ' && c != '\t' && c != '\r') return false;
	}
	return true;
}

// sort by (u, v); of several copies of one directed edge keep the last one in file order
void dedup_keep_last(std::vector<Edge>& e){
	std::stable_sort(e.begin(), e.end(), [](const Edge& a, const Edge& b){
		return a.u != b.u ? a.u < b.u : a.v < b.v;
	});
	size_t out = 0;
	for (size_t i = 0; i < e.size(); i++){
		if (i + 1 < e.size() && e[i].u == e[i + 1].u && e[i].v == e[i + 1].v) continue;
		e[out++] = e[i];
	}
	e.resize(out);
}

} // namespace

GraphData load_graph(const std::string& file, const LoadOptions& opt){
	std::ifstream in(file);
	if (!in) throw std::runtime_error("cannot open graph file: " + file);

	GraphData g;
	bool have_header = false;
	int hdr_n = -1, hdr_max = -1;
	int max_id = -1;
	const int shift = opt.one_indexed ? 1 : 0;

	std::string line;
	std::vector<std::string_view> tok;
	long ln = 0;
	while (std::getline(in, line)){
		ln++;
		tokenize(line, tok);
		if (tok.empty()) continue;
		if (tok[0] == "c") continue;                       // DIMACS comment line

		if (tok[0] == "p"){
			if (have_header) fail(file, ln, "duplicate header");
			if (!g.edges.empty()) fail(file, ln, "header must come before the edges");
			size_t k = 1;
			bool dimacs = (k < tok.size() && !is_number(tok[k]));   // 'p sp n m'
			if (dimacs) k++;
			if (k >= tok.size() || !to_int(tok[k], hdr_n) || hdr_n < 0){
				fail(file, ln, "bad vertex count in header");
			}
			k++;
			if (!dimacs && k < tok.size()){
				if (!to_int(tok[k], hdr_max) || hdr_max < hdr_n){
					fail(file, ln, "max vertices must be a number >= vertex count");
				}
			}
			have_header = true;
			continue;
		}

		if (tok[0] == "e"){
			fail(file, ln, "undirected edges ('e') are not supported; only directed arcs ('a' or 'u v [w]') are allowed");
		}

		size_t k = (tok[0] == "a") ? 1 : 0;
		if (tok.size() < k + 2 || tok.size() > k + 3){
			fail(file, ln, "expected: [a] u v [weight]");
		}
		int u, v, w = opt.default_weight;
		if (!to_int(tok[k], u) || !to_int(tok[k + 1], v)) fail(file, ln, "bad vertex id");
		if (tok.size() == k + 3 && !to_int(tok[k + 2], w)) fail(file, ln, "bad weight");
		u -= shift;
		v -= shift;
		if (u < 0 || v < 0) fail(file, ln, "negative vertex id");
		if (w < 0) fail(file, ln, "negative weight");
		if (have_header && (u >= hdr_n || v >= hdr_n)){
			fail(file, ln, "vertex id is not smaller than the vertex count");
		}
		max_id = std::max(max_id, std::max(u, v));
		g.edges.push_back(Edge{u, v, w});
	}

	g.n_v = have_header ? hdr_n : max_id + 1;
	int cap = opt.max_vertices > 0 ? opt.max_vertices
	                               : (hdr_max >= 0 ? hdr_max : g.n_v);
	if (cap < g.n_v) throw std::runtime_error(file + ": max vertices is smaller than vertex count");
	g.max_n_v = cap;

	dedup_keep_last(g.edges);
	return g;
}

void load_batch(const std::string& file, std::vector<HostBatch>& out, const LoadOptions& opt){
	std::ifstream in(file);
	if (!in) throw std::runtime_error("cannot open batch file: " + file);

	const int shift = opt.one_indexed ? 1 : 0;
	HostBatch cur;
	auto flush = [&](){
		if (!cur.ins.empty() || !cur.del.empty() || !cur.rem.empty()){
			out.push_back(std::move(cur));
		}
		cur = HostBatch();
	};

	std::string line;
	std::vector<std::string_view> tok;
	long ln = 0;
	while (std::getline(in, line)){
		ln++;
		if (is_blank(line)){ flush(); continue; }          // blank line ends a batch
		tokenize(line, tok);
		if (tok.empty()) continue;                         // comment-only line
		if (tok[0] == "---"){ flush(); continue; }

		if (tok[0].size() != 1){
			fail(file, ln, "unknown operation '" + std::string(tok[0]) + "'");
		}
		const char op = (char)std::toupper((unsigned char)tok[0][0]);

		auto id = [&](size_t k) -> int {
			int x;
			if (k >= tok.size() || !to_int(tok[k], x)) fail(file, ln, "missing or bad number");
			x -= shift;
			if (x < 0) fail(file, ln, "negative vertex id");
			return x;
		};

		switch (op){
		case 'I': {
			if (tok.size() < 3 || tok.size() > 4) fail(file, ln, "expected: I u v [weight]");
			int u = id(1), v = id(2), w = opt.default_weight;
			if (tok.size() == 4 && !to_int(tok[3], w)) fail(file, ln, "bad weight");
			if (w < 0) fail(file, ln, "negative weight");
			cur.ins.push_back(Edge{u, v, w});
			break;
		}
		case 'D': {
			if (tok.size() < 3 || tok.size() > 4) fail(file, ln, "expected: D u v");
			int u = id(1), v = id(2);
			cur.del.push_back(Edge{u, v, 0});
			break;
		}
		case 'R': {
			if (tok.size() != 2) fail(file, ln, "expected: R u");
			cur.rem.push_back(id(1));
			break;
		}
		default:
			fail(file, ln, "unknown operation '" + std::string(tok[0]) + "'");
		}
	}
	flush();
}
