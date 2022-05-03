set -ex

# Indexing
INDEX='python -m pya0.transformer_eval index ./utils/transformer_eval.ini'

$INDEX index_ntcir12_dpr --device titan_rtx
$INDEX index_ntcir12_dpr__3ep_pretrain_1ep --device titan_rtx
$INDEX index_ntcir12_dpr__7ep_pretrain_1ep --device titan_rtx
$INDEX index_ntcir12_dpr__scibert_1ep --device titan_rtx
$INDEX index_ntcir12_dpr__vanilla_1ep --device titan_rtx
$INDEX index_ntcir12_colbert --device titan_rtx

$INDEX index_arqmath2_dpr --device a6000_1
$INDEX index_arqmath2_dpr__3ep_pretrain_1ep --device a6000_1
$INDEX index_arqmath2_dpr__7ep_pretrain_1ep --device a6000_1
$INDEX index_arqmath2_dpr__scibert_1ep --device a6000_1
$INDEX index_arqmath2_dpr__vanilla_1ep --device a6000_1
$INDEX index_arqmath2_colbert --device a6000_1

# Searching
SEARCH='python -m pya0.transformer_eval search ./utils/transformer_eval.ini'

$SEARCH search_ntcir12_dpr --device cpu
$SEARCH search_ntcir12_dpr__3ep_pretrain_1ep --device cpu
$SEARCH search_ntcir12_dpr__7ep_pretrain_1ep --device cpu
$SEARCH search_ntcir12_dpr__scibert_1ep --device cpu
$SEARCH search_ntcir12_dpr__vanilla_1ep --device cpu
$SEARCH search_ntcir12_colbert --device a6000_1

$SEARCH search_arqmath2_dpr --device cpu
$SEARCH search_arqmath2_dpr__3ep_pretrain_1ep --device cpu
$SEARCH search_arqmath2_dpr__7ep_pretrain_1ep --device cpu
$SEARCH search_arqmath2_dpr__scibert_1ep --device cpu
$SEARCH search_arqmath2_dpr__vanilla_1ep --device cpu
$SEARCH search_arqmath2_colbert --device a6000_1

# Reranking
RERANK='python -m pya0.transformer_eval maprun ./utils/transformer_eval.ini'
baseline_run=./runs/arqmath2-a0-task1.run

$RERANK maprun_arqmath2_to_dpr $baseline_run --device a6000_1
$RERANK maprun_arqmath2_to_colbert $baseline_run --device a6000_1
