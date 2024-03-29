#pragma once

/* for MAX_TXT_SEG_BYTES */
#include "txt-seg/config.h"

/* mixed search debug */
//#define DEBUG_INDICES_RUN_QUERY

/* math search debug */
//#define DEBUG_PREPARE_MATH_QRY
//#define DEBUG_MATH_SEARCH
//#define DEBUG_MATH_SEARCH__STRUCT_SCORING
//#define DEBUG_MATH_SEARCH__SYMBOL_SCORING
//#define DEBUG_MATH_SEARCH__PRUNER_UPDATE

//#define IGNORE_MATH_SYMBOL_SCORE

#include "math-index-v3/config.h"
#define MAX_MATCHED_PATHS MAX_MATH_PATHS

// #define DEBUG_BIN_LP

// #define DEBUG_MATH_PRUNING

// #define DEBUG_MNC_SCORING

/* choose math pruning strategy */
//#define MATH_PRUNING_STRATEGY_NONE
//#define MATH_PRUNING_STRATEGY_MAXREF
//#define MATH_PRUNING_STRATEGY_GBP_NUM
  #define MATH_PRUNING_STRATEGY_GBP_LEN

//#define MATH_PRUNING_INIT_THRESHOLD_FACTOR .30f /* aggressive */
  #define MATH_PRUNING_INIT_THRESHOLD_FACTOR {{INIT_TH}} /* conservative */
//#define MATH_PRUNING_INIT_THRESHOLD_FACTOR .00f /* rank-safe */

#define MATH_PRUNING_NOTCLAUSE_THRESHOLD_FACTOR .60f /* for NOT clause */

#define MATH_SCORE_ETA {{MATH_SCORE_ETA}} /* larger: slow */

#define MATH_BASE_WEIGHT {{MATH_BASE_WEIGHT}}
#define MATH_REWARD_WEIGHT  (MATH_BASE_WEIGHT * 0.98f)
#define MATH_PENALTY_WEIGHT (MATH_BASE_WEIGHT * 0.02f)

#include "term-index/config.h" /* for MAX_TERM_ITEM_POSITIONS */

#define MAX_MATH_OCCURS 1
#define MAX_TOTAL_OCCURS (MAX_TERM_ITEM_POSITIONS + MAX_MATH_OCCURS * 2)

/* even a short query can get a long hilighted string, e.g.
 * in the case of math search */
#define MAX_HIGHLIGHTED_BYTES (MAX_TXT_SEG_BYTES * 64)

//#define DEBUG_SNIPPET_LEXER

#define SNIPPET_ROUGH_SZ    2048
#define SNIPPET_MAX_PADDING 768
#define SNIPPET_MIN_PADDING 128
#define MAX_SNIPPET_SZ \
	(MAX_TOTAL_OCCURS * (MAX_TXT_SEG_BYTES + (SNIPPET_MAX_PADDING + 1) * 2))

#define DEFAULT_K_TOP_RESULTS 20
#define MAX_K_TOP_RESULTS   1000
#define DEFAULT_RES_PER_PAGE  10

{{SYMBOL_SCORING_ENABLE}} SYMBOL_SCORING_ENABLE
#define SYMBOL_SUBSCORE_FULL 1.00f /* leaf and fingerprint match */
#define SYMBOL_SUBSCORE_LEAF {{SYMBOL_SUBSCORE_LEAF}} /* only leaf match */
#define SYMBOL_SUBSCORE_BASE {{SYMBOL_SUBSCORE_BASE}} /* base score */

{{PATH_IDF_ENABLE}} PATH_IDF_ENABLE

#define BM25_DEFAULT_B  0.75f
#define BM25_DEFAULT_K1 2.0f

#define MAX_QUERY_BYTES     (MAX_TXT_SEG_BYTES * 32)
#define MAX_QUERY_WSTR_LEN  (MAX_TXT_SEG_LEN * 32)

#define PRINT_SEARCH_QUERIES

#define PRINT_MERGE_TIME

#define SEARCH_MERGER__USE_MAXSCORE
//#define SEARCH_MERGER__USE_WAND

#define UNLIMIT_FILENO

#define EFFICIENT_LOCAL_THRESHOLD_UPDATE
