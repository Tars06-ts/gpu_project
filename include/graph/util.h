#ifndef UTIL_H
#define UTIL_H

#define MIN_SLACK 20

inline int next_power_of_2(int u){
    if (u <= 1) return 1;
    // Prevent signed overflow for values > 2^30 (1,073,741,824)
    if (u > 1073741824) return 1073741824; 

    unsigned int v = static_cast<unsigned int>(u - 1);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return static_cast<int>(v + 1);	
}

template <typename... Args>
inline int max(int first, Args... rest) {
    int res = first;
    ((res = (rest > res) ? rest : res), ...);
    return res;
}

template <typename... Args>
inline int min(int first, Args... rest) {
    int res = first;
    ((res = (rest < res) ? rest : res), ...);
    return res;
}

#endif
