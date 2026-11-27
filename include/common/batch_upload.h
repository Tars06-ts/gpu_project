#ifndef BATCH_UPLOAD_H
#define BATCH_UPLOAD_H
#include "common/types.h"

// copies a parsed batch to the device (empty lists give nullptr)
Batch upload_batch(const HostBatch& h);
// frees the device arrays and resets the counts
void free_batch(Batch& b);

#endif
