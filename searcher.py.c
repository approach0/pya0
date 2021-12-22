#include <string.h>

#include "common/common.h"
#include "txt-seg/txt-seg.h"
#include "txt-seg/lex.h"
#include "search-v3/search.h"
#include "searchd/json-utils.h"
#include "head.h"

PyObject *do_search(PyObject *self, PyObject *args, PyObject* kwargs)
{
	/* parse arguments */
	PyObject *pyindices, *pylist;
	int verbose = 0, topk = 20;
	unsigned int docid = 0;
	const char *log = NULL;
	static char *kwlist[] = {
		"index", "keywords", "verbose", "topk", "log", "docid", NULL
	};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|pizI", kwlist,
		&pyindices, &pylist, &verbose, &topk, &log, &docid)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTupleAndKeywords error");
		return NULL;
	}

	/* sanity check */
	struct indices *indices = PyLong_AsVoidPtr(pyindices);
	int list_len = PyObject_Length(pylist);
	if (list_len <= 0) {
		PyErr_Format(PyExc_RuntimeError,
			"Please pass a valid list of keywords as query");
		return NULL;
	} else if (indices == NULL) {
		PyErr_Format(PyExc_RuntimeError,
			"Cannot open an empty index");
		return NULL;
	}

	/* construct query */
	struct query qry = QUERY_NEW;
	qry.docID = docid;
	for (int i = 0; i < list_len; i++) {
		PyObject *item = PyList_GetItem(pylist, i);
		if (!PyDict_Check(item)) {
			PyErr_Format(PyExc_RuntimeError,
				"Query list should contain a dictionary object");
			return NULL;
		}

		PyObject *py_kw = PyDict_GetItemString(item, "str");
		PyObject *py_type = PyDict_GetItemString(item, "type");
		PyObject *py_field = PyDict_GetItemString(item, "field");
		PyObject *py_boost = PyDict_GetItemString(item, "boost");

		if (py_kw == NULL || py_type == NULL) {
			PyErr_Format(PyExc_RuntimeError,
				"Required key(s) not found in query keyword");
			return NULL;
		}

		const char *kw_str = PyUnicode_AsUTF8(py_kw);
		const char *type_str = PyUnicode_AsUTF8(py_type);
		const char *field = py_field ? PyUnicode_AsUTF8(py_field) : "content";
		float boost = (py_boost == NULL) ? 1.f : PyFloat_AS_DOUBLE(py_boost);

		if (0 == strcmp(type_str, "term")) {
			/* lookup lexer for this field */
			lexer_handler lexer = NULL;
			lexer = indices_field_lexer(indices, "content");
			query_digest_txt(&qry, field, kw_str,
			                 QUERY_OP_OR, boost, lexer);

		} else if (0 == strcmp(type_str, "tex")) {
			query_push_kw(&qry, field, kw_str,
			              QUERY_KW_TEX, QUERY_OP_OR, boost);

		} else {
			PyErr_Format(PyExc_RuntimeError,
				"Bad query keyword type");
			return NULL;
		}
	}

	/* print query in verbose mode */
	if (verbose)
		query_print(qry, stdout);

	/* actually perform search */
	ranked_results_t srch_res; /* search results */
	if (verbose) {
		srch_res = indices_run_query(indices, &qry, topk, NULL, stdout);
	} else {
		FILE *log_fh = (log == NULL) ? fopen("/dev/null", "a") : fopen(log, "a");
		srch_res = indices_run_query(indices, &qry, topk, NULL, log_fh);
		fclose(log_fh);
	}

	/* convert search results to JSON stringified */
	const char *ret = search_results_json(&srch_res, -1, 0, indices);

	/* release resources */
	free_ranked_results(&srch_res);
	query_delete(qry);
	return PyUnicode_FromString(ret);
}
