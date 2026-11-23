#ifndef TYPES_H
#define TYPES_H

struct Edge{
    int u, v, weight;
};

struct Neighbor{
    int v, weight;
};

// insertion batch (all device pointers)
struct InsertCtx {
	Edge *ins;  int n_ins;
	bool *pool, *new_vertex, *is_update;
	int  *out_needed, *in_needed, *out_slots, *in_slots;
};


#endif
