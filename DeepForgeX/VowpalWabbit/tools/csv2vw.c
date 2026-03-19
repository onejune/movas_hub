/*
 * csv2vw - Fast CSV to VW format converter
 * 
 * Compile: gcc -O3 -march=native -o csv2vw csv2vw.c
 * Usage: ./csv2vw schema.txt colnames.txt label_idx key_idx < input.csv > output.vw
 *
 * Delimiter: \x02 (ASCII 2)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 1048576      // 1MB per line
#define MAX_COLS 2000
#define MAX_FEATURES 500
#define MAX_CROSS 100
#define MAX_CROSS_DEPTH 5
#define DELIMITER '\x02'

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

SingleFeature single_features[MAX_FEATURES];
int num_single = 0;

CrossFeature cross_features[MAX_CROSS];
int num_cross = 0;

// Column pointers for current line
char *columns[MAX_COLS];
int num_columns = 0;

// Column name to index mapping (simple hash table)
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

// Load column names
void load_colnames(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    
    char line[256];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        int idx;
        char name[64];
        if (sscanf(line, "%d %63s", &idx, name) == 2) {
            // colnames is 0-indexed
            hash_insert(name, idx);
            count++;
        }
    }
    fclose(f);
    fprintf(stderr, "Loaded %d column names\n", count);
}

// Load schema (combine_schema)
void load_schema(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    
    char line[512];
    char line_copy[512];  // strtok modifies the string
    while (fgets(line, sizeof(line), f)) {
        // Trim
        char *p = line;
        while (*p && isspace(*p)) p++;
        char *end = p + strlen(p) - 1;
        while (end > p && isspace(*end)) *end-- = 0;
        if (!*p) continue;
        
        // Make a copy for strtok
        strncpy(line_copy, p, sizeof(line_copy) - 1);
        line_copy[sizeof(line_copy) - 1] = 0;
        
        // Check for cross feature (contains #)
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

// Split line into columns
int split_line(char *line) {
    num_columns = 0;
    char *p = line;
    columns[num_columns++] = p;
    
    while (*p) {
        if (*p == DELIMITER) {
            *p = 0;
            if (num_columns < MAX_COLS) {
                columns[num_columns++] = p + 1;
            }
        }
        p++;
    }
    return num_columns;
}

// Sanitize value (inline)
int sanitize_value(const char *val, char *out, int max_len) {
    if (!val || !*val || strcmp(val, "-") == 0 || 
        strcmp(val, "-1.0") == 0 || strcmp(val, "-1") == 0) {
        return 0;  // NULL
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

// Output buffer
char output_buf[MAX_LINE * 2];
char *out_ptr;

void out_str(const char *s) {
    while (*s) *out_ptr++ = *s++;
}

void out_char(char c) {
    *out_ptr++ = c;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s schema.txt colnames.txt label_idx key_idx < input.csv\n", argv[0]);
        return 1;
    }
    
    const char *schema_path = argv[1];
    const char *colnames_path = argv[2];
    int label_idx = atoi(argv[3]);
    int key_idx = atoi(argv[4]);
    
    // Init hash table
    memset(hash_table, 0, sizeof(hash_table));
    
    // Load config
    load_colnames(colnames_path);
    load_schema(schema_path);
    
    // Process stdin
    char *line = malloc(MAX_LINE);
    char val_buf[256];
    long long row_count = 0;
    
    while (fgets(line, MAX_LINE, stdin)) {
        // Remove trailing newline
        int len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = 0;
        if (len > 1 && line[len-2] == '\r') line[len-2] = 0;
        
        split_line(line);
        
        // Label
        int label = -1;
        if (label_idx < num_columns) {
            double v = atof(columns[label_idx]);
            label = (v == 1.0) ? 1 : -1;
        }
        
        // Start output
        out_ptr = output_buf;
        
        // Label
        if (label == 1) {
            out_char('1');
        } else {
            out_str("-1");
        }
        
        // Tag (business_type)
        if (key_idx >= 0 && key_idx < num_columns && columns[key_idx][0]) {
            out_str(" '");
            out_str(columns[key_idx]);
        }
        
        // Single features
        int has_single = 0;
        for (int i = 0; i < num_single; i++) {
            int idx = single_features[i].col_idx;
            if (idx < num_columns && sanitize_value(columns[idx], val_buf, sizeof(val_buf))) {
                if (!has_single) {
                    out_str(" |s");
                    has_single = 1;
                }
                out_char(' ');
                out_str(single_features[i].name);
                out_char('=');
                out_str(val_buf);
            }
        }
        
        // Cross features
        int has_cross = 0;
        char cross_name[256], cross_val[512];
        
        for (int i = 0; i < num_cross; i++) {
            CrossFeature *cf = &cross_features[i];
            int valid = 1;
            
            // Check all values exist
            char *vals[MAX_CROSS_DEPTH];
            char val_bufs[MAX_CROSS_DEPTH][256];
            
            for (int j = 0; j < cf->count; j++) {
                int idx = cf->col_indices[j];
                if (idx >= num_columns || !sanitize_value(columns[idx], val_bufs[j], sizeof(val_bufs[j]))) {
                    valid = 0;
                    break;
                }
                vals[j] = val_bufs[j];
            }
            
            if (valid) {
                if (!has_cross) {
                    out_str(" |c");
                    has_cross = 1;
                }
                
                // Build cross feature name and value
                out_char(' ');
                for (int j = 0; j < cf->count; j++) {
                    if (j > 0) out_str("_X_");
                    out_str(cf->names[j]);
                }
                out_char('=');
                for (int j = 0; j < cf->count; j++) {
                    if (j > 0) out_str("_X_");
                    out_str(vals[j]);
                }
            }
        }
        
        out_char('\n');
        *out_ptr = 0;
        
        fputs(output_buf, stdout);
        
        row_count++;
        if (row_count % 1000000 == 0) {
            fprintf(stderr, "Processed %lld rows...\n", row_count);
        }
    }
    
    fprintf(stderr, "Total: %lld rows\n", row_count);
    free(line);
    return 0;
}
