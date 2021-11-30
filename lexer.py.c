#include <stdio.h>
#include <stdlib.h>

#include "head.h"
#include "tex-parser/head.h"
#include "tex-parser/y.tab.h"

extern size_t lex_cur_bytes;
extern int yyleng;

PyObject *do_lexing(PyObject *self, PyObject *args, PyObject* kwargs)
{
	char *string;
	int include_syntatic_literal = 0, include_spans = 0;
	static char *kwlist[] = {"latex",
        "include_syntatic_literal", "include_spans", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|pp", kwlist,
		&string, &include_syntatic_literal, &include_spans)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTupleAndKeywords error");
		return NULL;
	}

	/* create parser buffer */
	size_t scan_buf_sz;
	char *scan_buf;
	scan_buf = mk_scan_buf(string, &scan_buf_sz);
	YY_BUFFER_STATE state_buf;
	state_buf = yy_scan_buffer(scan_buf, scan_buf_sz);

	/* allocate Python List */
	PyObject *item, *list = PyList_New(0);

	/* scan tokens */
	char *token = NULL, *symbol = NULL;
	int next;
	yylval.nd = NULL; /* FIX: avoid non-NULL */
	while ((next = yylex())) {
		struct optr_node* nd = yylval.nd;
		int span[2] = {lex_cur_bytes - yyleng, lex_cur_bytes};
		if (nd) {
			/* get token and symbol in string */
			token = trans_token(nd->token_id);
			symbol = trans_symbol(nd->symbol_id);
			/* append item */
			if (include_spans)
				item = Py_BuildValue("lss(ii)",
					next, token, symbol, span[0], span[1]);
			else
				item = Py_BuildValue("lss", next, token, symbol);
			PyList_Append(list, item); /* only lend the ref */
			Py_DECREF(item);
			/* release union */
			optr_release(nd);
			yylval.nd = NULL;
		} else if (include_syntatic_literal) {
			Py_INCREF(Py_None);
			if (include_spans)
				item = Py_BuildValue("lOs(ii)",
					next, Py_None, yytext, span[0], span[1]);
			else
				item = Py_BuildValue("lOs", next, Py_None, yytext);
			PyList_Append(list, item); /* only lend the ref */
			Py_DECREF(item);
        }
	}

	yy_delete_buffer(state_buf);
	free(scan_buf);

	yylex_destroy();
	return list;
}

PyObject *use_fallback_parser(PyObject *self, PyObject *args)
{
	int use_fallback = 0;
	if (!PyArg_ParseTuple(args, "p", &use_fallback)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTuple error");
		return NULL;
	}

	tex_parser_use_fallback(use_fallback);
	Py_RETURN_NONE;
}
