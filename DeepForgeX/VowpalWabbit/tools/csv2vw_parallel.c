/*
 * csv2vw_parallel - Parallel CSV to VW format converter
 * 
 * Compile: gcc -O3 -march=native -pthread -o csv2vw_parallel csv2vw_parallel.c
 * Usage: ./csv2vw_parallel schema.txt colnames.txt label_idx key_idx num_threads < input.csv > output.vw
 *
 * Delimiter: \x02 (ASCII 2)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>
#include <unistd.h>

#define MAX_LINE 1048576      // 1MB per line
#define MAX_COLS 2000
#define MAX_FEATURES 500
#define MAX_CROSS 150
#define MAX_CROSS_DEPTH 5
#define DELIMITER '\x02'
#define BATCH_SIZE 10000      // Lines per batch

// Feature structures
typedef struct {
    char name[64];
    int col_idx;
} SingleFeature;

typedef struct {
    char names[MAX_CROSS_DEPTH][64];
    int col_indices[MAX_CROSS_DEPTH];
    int count;
} CrossFeature;

// Global config (read-only after init)
SingleFeature single_features[MAX_FEATURES];
int num_single = 0;
CrossFeature cross_features[MAX_CROSS];
int num_cross = 0;
int g_label_idx = 0;
int g_key_idx = 0;

// Column name to index mapping
#define HASH_SIZE 4096
typedef struct HashEntry {
    char name[64];
    int idx;
    struct HashEntry *next;
} HashEntry;
HashEntry *hash_table[HASH_SIZE];

unsigned int hash_str(const char *s) {
    unsigned int h = 0;
    while (*s) h = h * 31 + (unsigned char)*s++;
    return h % HASH_SIZE;
}

void hash_insert(const char *name, int idx) {
    unsigned int h = hash_str(name);
    HashEntry *e = malloc(sizeof(HashEntry));
    strncpy(e->name, name, 63);
    e->name[63] = 0;
    e->idx = idx;
    e->next = hash_table[h];
    hash_table[h] = e;
}

int hash_lookup(const char *name) {
    unsigned int h = hash_str(name);
    for (HashEntry *e = hash_table[h]; e; e = e->next) {
        if (strcmp(e->name, name) == 0) return e->idx;
    }
    return -1;
}

void load_colnames(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    char line[256];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        int idx;
        char name[64];
        if (sscanf(line, "%d %63s", &idx, name) == 2) {
            hash_insert(name, idx);
            count++;
        }
    }
    fclose(f);
    fprintf(stderr, "Loaded %d column names\n", count);
}

void load_schema(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    char line[512], line_copy[512];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p && isspace(*p)) p++;
        char *end = p + strlen(p) - 1;
        while (end > p && isspace(*end)) *end-- = 0;
        if (!*p) continue;
        strncpy(line_copy, p, sizeof(line_copy) - 1);
        line_copy[sizeof(line_copy) - 1] = 0;
        
        if (strchr(p, '#')) {
            if (num_cross >= MAX_CROSS) continue;
            CrossFeature *cf = &cross_features[num_cross];
            cf->count = 0;
            char *tok = strtok(line_copy, "#");
            while (tok && cf->count < MAX_CROSS_DEPTH) {
                int idx = hash_lookup(tok);
                if (idx >= 0) {
                    strncpy(cf->names[cf->count], tok, 63);
                    cf->names[cf->count][63] = 0;
                    cf->col_indices[cf->count] = idx;
                    cf->count++;
                }
                tok = strtok(NULL, "#");
            }
            if (cf->count >= 2) num_cross++;
        } else {
            if (num_single >= MAX_FEATURES) continue;
            int idx = hash_lookup(p);
            if (idx >= 0) {
                strncpy(single_features[num_single].name, p, 63);
                single_features[num_single].name[63] = 0;
                single_features[num_single].col_idx = idx;
                num_single++;
            }
        }
    }
    fclose(f);
    fprintf(stderr, "Loaded %d single features, %d cross features\n", num_single, num_cross);
}

// Thread-local processing
typedef struct {
    char **lines;           // Input lines
    char **outputs;         // Output lines
    int count;              // Number of lines
    int thread_id;
} WorkBatch;

int sanitize_value(const char *val, char *out, int max_len) {
    if (!val || !*val || strcmp(val, "-") == 0 || 
        strcmp(val, "-1.0") == 0 || strcmp(val, "-1") == 0) {
        return 0;
    }
    int i = 0;
    while (*val && i < max_len - 1) {
        char c = *val++;
        if (c == ' ' || c == ':' || c == '|' || c == '\t' || c == '\n' || c == '\r') {
            out[i++] = '_';
        } else {
            out[i++] = c;
        }
    }
    out[i] = 0;
    return i > 0;
}

void process_line(const char *input, char *output, int max_out) {
    char line[MAX_LINE];
    strncpy(line, input, MAX_LINE - 1);
    line[MAX_LINE - 1] = 0;
    
    // Remove trailing newline
    int len = strlen(line);
    if (len > 0 && line[len-1] == '\n') line[--len] = 0;
    if (len > 0 && line[len-1] == '\r') line[--len] = 0;
    
    // Split into columns
    char *columns[MAX_COLS];
    int num_columns = 0;
    char *p = line;
    columns[num_columns++] = p;
    while (*p) {
        if (*p == DELIMITER) {
            *p = 0;
            if (num_columns < MAX_COLS) columns[num_columns++] = p + 1;
        }
        p++;
    }
    
    // Build output
    char *out = output;
    char *out_end = output + max_out - 1;
    char val_buf[256];
    
    // Label
    int label = -1;
    if (g_label_idx < num_columns) {
        double v = atof(columns[g_label_idx]);
        label = (v == 1.0) ? 1 : -1;
    }
    
    if (label == 1) {
        *out++ = '1';
    } else {
        *out++ = '-'; *out++ = '1';
    }
    
    // Tag
    if (g_key_idx >= 0 && g_key_idx < num_columns && columns[g_key_idx][0]) {
        *out++ = ' '; *out++ = '\'';
        const char *tag = columns[g_key_idx];
        while (*tag && out < out_end) *out++ = *tag++;
    }
    
    // Single features
    int has_single = 0;
    for (int i = 0; i < num_single && out < out_end - 256; i++) {
        int idx = single_features[i].col_idx;
        if (idx < num_columns && sanitize_value(columns[idx], val_buf, sizeof(val_buf))) {
            if (!has_single) {
                *out++ = ' '; *out++ = '|'; *out++ = 's';
                has_single = 1;
            }
            *out++ = ' ';
            const char *name = single_features[i].name;
            while (*name) *out++ = *name++;
            *out++ = '=';
            const char *val = val_buf;
            while (*val) *out++ = *val++;
        }
    }
    
    // Cross features
    int has_cross = 0;
    char val_bufs[MAX_CROSS_DEPTH][256];
    
    for (int i = 0; i < num_cross && out < out_end - 512; i++) {
        CrossFeature *cf = &cross_features[i];
        int valid = 1;
        
        for (int j = 0; j < cf->count; j++) {
            int idx = cf->col_indices[j];
            if (idx >= num_columns || !sanitize_value(columns[idx], val_bufs[j], sizeof(val_bufs[j]))) {
                valid = 0;
                break;
            }
        }
        
        if (valid) {
            if (!has_cross) {
                *out++ = ' '; *out++ = '|'; *out++ = 'c';
                has_cross = 1;
            }
            *out++ = ' ';
            
            // Feature name
            for (int j = 0; j < cf->count; j++) {
                if (j > 0) { *out++ = '_'; *out++ = 'X'; *out++ = '_'; }
                const char *name = cf->names[j];
                while (*name) *out++ = *name++;
            }
            *out++ = '=';
            // Feature value
            for (int j = 0; j < cf->count; j++) {
                if (j > 0) { *out++ = '_'; *out++ = 'X'; *out++ = '_'; }
                const char *val = val_bufs[j];
                while (*val) *out++ = *val++;
            }
        }
    }
    
    *out++ = '\n';
    *out = 0;
}

void *worker_thread(void *arg) {
    WorkBatch *batch = (WorkBatch *)arg;
    
    for (int i = 0; i < batch->count; i++) {
        batch->outputs[i] = malloc(MAX_LINE * 2);
        process_line(batch->lines[i], batch->outputs[i], MAX_LINE * 2);
    }
    
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s schema.txt colnames.txt label_idx key_idx num_threads < input.csv\n", argv[0]);
        return 1;
    }
    
    const char *schema_path = argv[1];
    const char *colnames_path = argv[2];
    g_label_idx = atoi(argv[3]);
    g_key_idx = atoi(argv[4]);
    int num_threads = atoi(argv[5]);
    
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 64) num_threads = 64;
    
    fprintf(stderr, "Using %d threads\n", num_threads);
    
    memset(hash_table, 0, sizeof(hash_table));
    load_colnames(colnames_path);
    load_schema(schema_path);
    
    // Allocate batches
    WorkBatch *batches = malloc(num_threads * sizeof(WorkBatch));
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    
    for (int i = 0; i < num_threads; i++) {
        batches[i].lines = malloc(BATCH_SIZE * sizeof(char *));
        batches[i].outputs = malloc(BATCH_SIZE * sizeof(char *));
        batches[i].thread_id = i;
    }
    
    char *line_buf = malloc(MAX_LINE);
    long long total_rows = 0;
    int lines_in_batch = 0;
    int current_batch = 0;
    
    // Read and process
    while (fgets(line_buf, MAX_LINE, stdin)) {
        batches[current_batch].lines[lines_in_batch] = strdup(line_buf);
        lines_in_batch++;
        
        // When batch is full, assign to next thread
        if (lines_in_batch >= BATCH_SIZE / num_threads) {
            batches[current_batch].count = lines_in_batch;
            current_batch++;
            lines_in_batch = 0;
            
            // When all threads have work, process in parallel
            if (current_batch >= num_threads) {
                // Launch threads
                for (int i = 0; i < num_threads; i++) {
                    pthread_create(&threads[i], NULL, worker_thread, &batches[i]);
                }
                
                // Wait and output in order
                for (int i = 0; i < num_threads; i++) {
                    pthread_join(threads[i], NULL);
                    for (int j = 0; j < batches[i].count; j++) {
                        fputs(batches[i].outputs[j], stdout);
                        free(batches[i].outputs[j]);
                        free(batches[i].lines[j]);
                        total_rows++;
                    }
                }
                
                current_batch = 0;
                
                if (total_rows % 1000000 == 0) {
                    fprintf(stderr, "Processed %lld rows...\n", total_rows);
                }
            }
        }
    }
    
    // Process remaining lines
    if (lines_in_batch > 0) {
        batches[current_batch].count = lines_in_batch;
        current_batch++;
    }
    
    if (current_batch > 0) {
        for (int i = 0; i < current_batch; i++) {
            pthread_create(&threads[i], NULL, worker_thread, &batches[i]);
        }
        for (int i = 0; i < current_batch; i++) {
            pthread_join(threads[i], NULL);
            for (int j = 0; j < batches[i].count; j++) {
                fputs(batches[i].outputs[j], stdout);
                free(batches[i].outputs[j]);
                free(batches[i].lines[j]);
                total_rows++;
            }
        }
    }
    
    fprintf(stderr, "Total: %lld rows\n", total_rows);
    
    // Cleanup
    for (int i = 0; i < num_threads; i++) {
        free(batches[i].lines);
        free(batches[i].outputs);
    }
    free(batches);
    free(threads);
    free(line_buf);
    
    return 0;
}
