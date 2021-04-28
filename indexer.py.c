#include <string.h>

#include "txt-seg/txt-seg.h"
#include "txt-seg/lex.h"
#include "indices-v3/indices.h"
#include "head.h"

PyObject *index_open(PyObject *self, PyObject *args, PyObject* kwargs)
{
	const char *path, *option = NULL, *seg_dict = NULL;
	static char *kwlist[] = {"path", "option", "segment_dict", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|ss", kwlist,
		&path, &option, &seg_dict)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTupleAndKeywords error");
		return NULL;
	}

	int failed;
	struct indices *indices = malloc(sizeof *indices);
	if (NULL == option || NULL != strstr(option, "w")) {
		failed = indices_open(indices, path, INDICES_OPEN_RW);
	} else {
		failed = indices_open(indices, path, INDICES_OPEN_RD);
	}

	if (failed) {
		free(indices);
		Py_INCREF(Py_None); // return a new reference of Py_None
		return Py_None;
	}

	struct indices_field fields[] = {
		{"url", FIELD__STORE_PLAIN, FIELD__INDEX_NO},
		{"content", FIELD__STORE_COMPRESSED, FIELD__INDEX_YES},
		{"extern_id", FIELD__STORE_PLAIN, FIELD__INDEX_NO}
	};
	(void)indices_schema_add_field(indices, fields,
		sizeof(fields) / sizeof(fields[0]));

	if (seg_dict)
		text_segment_init(seg_dict);

	return PyLong_FromVoidPtr(indices);
}

PyObject *index_close(PyObject *self, PyObject *args)
{
	PyObject *pyindices;
	if (!PyArg_ParseTuple(args, "O", &pyindices)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTuple error");
		return NULL;
	}

	struct indices *indices = PyLong_AsVoidPtr(pyindices);
	indices_close(indices);
	free(indices);
	text_segment_free();

	Py_RETURN_NONE;
}

PyObject *index_memcache(PyObject *self, PyObject *args, PyObject* kwargs)
{
	PyObject *pyindices;
	int term_cache = 0, math_cache = 0; /* in MiB */
	static char *kwlist[] = {"index", "term_cache", "math_cache", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ii", kwlist,
		&pyindices, &term_cache, &math_cache)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTupleAndKeywords error");
		return NULL;
	}

	struct indices *indices = PyLong_AsVoidPtr(pyindices);
	indices->ti_cache_limit = term_cache MB;
	indices->mi_cache_limit = math_cache MB;
	indices_cache(indices);

	Py_RETURN_NONE;
}

PyObject *index_print_summary(PyObject *self, PyObject *args)
{
	PyObject *pyindices;
	if (!PyArg_ParseTuple(args, "O", &pyindices)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTuple error");
		return NULL;
	}

	struct indices *indices = PyLong_AsVoidPtr(pyindices);
	indices_print_summary(indices);

	Py_RETURN_NONE;
}

PyObject *index_lookup_doc(PyObject *self, PyObject *args)
{
	unsigned int key;
	PyObject *pyindices;
	if (!PyArg_ParseTuple(args, "OI", &pyindices, &key)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTuple error");
		return NULL;
	}

	if (key == 0) {
		PyErr_Format(PyExc_RuntimeError,
			"key#0 never exists");
		return NULL;
    }

	struct indices *indices = PyLong_AsVoidPtr(pyindices);
	int n_field = indices->n_field;
	struct indices_field *fields = indices->fields;

	PyObject *result = PyDict_New();
	for (int i = 0; i < n_field; i++) {
		if (!fields[i].store)
			continue;
		char *field_name = fields[i].name;
		bool compress = (fields[i].store == FIELD__STORE_COMPRESSED);
		size_t len;
		char *s = get_blob_txt(indices->bi[i], key, compress, &len);
		if (s) {
			// dict setter does NOT steal the reference
			PyObject* py_s = PyUnicode_FromString(s);
			PyDict_SetItemString(result, field_name, py_s);
			Py_DECREF(py_s);
			free(s);
		} else {
			// dict setter does NOT steal the reference
			PyObject* py_s = PyUnicode_FromString("");
			PyDict_SetItemString(result, field_name, py_s);
			Py_DECREF(py_s);
		}
	}

	return result;
}

static int
parser_exception(struct indexer *indexer, const char *tex, char *msg)
{
	fprintf(stderr, "Parser error: %s in `%s'\n", msg, tex);
	return 0;
}

PyObject *indexer_new(PyObject *self, PyObject *args)
{
	PyObject *pyindices;
	if (!PyArg_ParseTuple(args, "O", &pyindices)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTuple error");
		return NULL;
	}

	struct indices *indices = PyLong_AsVoidPtr(pyindices);
	struct indexer *indexer;
	indexer = indexer_alloc(indices, INDICES_TXT_LEXER, parser_exception);

	return PyLong_FromVoidPtr(indexer);
}

PyObject *indexer_del(PyObject *self, PyObject *args)
{
	PyObject *pyindexer;
	if (!PyArg_ParseTuple(args, "O", &pyindexer)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTuple error");
		return NULL;
	}

	struct indexer *indexer = PyLong_AsVoidPtr(pyindexer);
	indexer_free(indexer);

	Py_RETURN_NONE;
}

PyObject *do_maintain(PyObject *self, PyObject *args, PyObject* kwargs)
{
	PyObject *pyindexer;
	int force = 0;
	static char *kwlist[] = {"writer", "force", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist,
		&pyindexer, &force)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTupleAndKeywords error");
		return NULL;
	}

	struct indexer *indexer = PyLong_AsVoidPtr(pyindexer);
	if (indexer_should_maintain(indexer) || force) {
		indexer_maintain(indexer);
		return PyBool_FromLong(1);
	} else {
		return PyBool_FromLong(0);
	}
}

PyObject *do_flush(PyObject *self, PyObject *args)
{
	PyObject *pyindexer;
	if (!PyArg_ParseTuple(args, "O", &pyindexer)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTuple error");
		return NULL;
	}

	struct indexer *indexer = PyLong_AsVoidPtr(pyindexer);
	indexer_flush(indexer);

	Py_RETURN_NONE;
}

PyObject *add_document(PyObject *self, PyObject *args, PyObject* kwargs)
{
	PyObject *pyindexer;
	const char *content, *url = NULL, *extern_id_str = NULL;
	static char *kwlist[] = {"writer", "content", "url", "extern_id", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Os|ss", kwlist,
		&pyindexer, &content, &url, &extern_id_str)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTupleAndKeywords error");
		return NULL;
	}

	struct indexer *indexer = PyLong_AsVoidPtr(pyindexer);
	struct indices *indices = indexer->indices;
	uint32_t docID = indices->n_doc + 1;

	bool anyfield_written = false;

	if (extern_id_str) {
		uint32_t extern_id;
		if (1 == sscanf(extern_id_str, "%lu", &extern_id)) {
			char docIDstr[1024];
			snprintf(docIDstr, 1024, "%d", docID);
			(void)indexer_write_field(indexer, extern_id, "extern_id", docIDstr);
			anyfield_written = true;
		}
	}

	if (content) {
		/* for all the other fields */
		(void)indexer_write_field(indexer, 0, "content", content);
		anyfield_written = true;
	} else {
		return PyLong_FromUnsignedLong(0);
	}

	if (url) {
		/* for all the other fields */
		(void)indexer_write_field(indexer, 0, "url", url);
		anyfield_written = true;
	}

	if (anyfield_written) {
		/* maintain and prepare for the next indexing document */
		docID = indexer_next_doc(indexer);
	} else {
		docID = 0;
	}

	return PyLong_FromUnsignedLong(docID);
}
