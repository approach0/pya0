#include <stdio.h>
#include <stdlib.h>

#include "head.h"
#include "tex-parser/head.h"

static PyObject *get_opt_pyobject(struct optr_node*, int);

struct append_subtree_pyobject_args {
	PyObject *py_list;
	int insert_rank_node;
	int commutative;
	int rank;
};

static LIST_IT_CALLBK(append_subtree_pyobject)
{
	TREE_OBJ(struct optr_node, opt, tnd);
	P_CAST(args, struct append_subtree_pyobject_args, pa_extra);
	PyObject *subtree = get_opt_pyobject(opt, args->insert_rank_node);

	if (args->insert_rank_node && !args->commutative) {
		PyObject *rank = PyTuple_New(6);
		PyObject *span = PyTuple_New(2);

		PyTuple_SetItem(span, 0, PyLong_FromLong(0));
		PyTuple_SetItem(span, 1, PyLong_FromLong(0));

		PyObject *py_list = PyList_New(0);
		PyList_Append(py_list, subtree);

		PyTuple_SetItem(rank, 0, PyLong_FromLong(opt->node_id));
		PyTuple_SetItem(rank, 1, PyLong_FromLong(opt->sign));
		PyTuple_SetItem(rank, 2, PyUnicode_FromString("RANK"));
		PyTuple_SetItem(rank, 3, PyUnicode_FromFormat("rank_%d", args->rank));
		PyTuple_SetItem(rank, 4, span);
		PyTuple_SetItem(rank, 5, py_list);
		PyList_Append(args->py_list, rank);

	} else {
		PyList_Append(args->py_list, subtree);
	}

	args->rank += 1;
	LIST_GO_OVER;
}

static PyObject *get_opt_pyobject(struct optr_node *opt, int insert_rank_node)
{
	if (opt == NULL) {
		Py_RETURN_NONE;
	}

	PyObject *py_list = PyList_New(0);
	struct append_subtree_pyobject_args args = {
		py_list, insert_rank_node, opt->commutative, 1
	};
	list_foreach(&opt->tnd.sons, &append_subtree_pyobject, &args);

	char *token = trans_token(opt->token_id);
	char *symbol = trans_symbol(opt->symbol_id);

	/* prepend a start at the symbol of wildcards */
	if (opt->wildcard) {
		size_t l = strlen(symbol);
		symbol[l + 1] = '\0';
		for (size_t i = l; i >= 1; i--) {
			symbol[i] = symbol[i - 1];
		}
		symbol[0] = '*';
	}


	PyObject *result = PyTuple_New(6);
	PyObject *span = PyTuple_New(2);

	PyTuple_SetItem(span, 0, PyLong_FromLong(opt->pos_begin));
	PyTuple_SetItem(span, 1, PyLong_FromLong(opt->pos_end));

	PyTuple_SetItem(result, 0, PyLong_FromLong(opt->node_id));
	PyTuple_SetItem(result, 1, PyLong_FromLong(opt->sign));
	PyTuple_SetItem(result, 2, PyUnicode_FromString(token));
	PyTuple_SetItem(result, 3, PyUnicode_FromString(symbol));
	PyTuple_SetItem(result, 4, span);
	PyTuple_SetItem(result, 5, py_list);
	return result;
}

PyObject *do_parsing(PyObject *self, PyObject *args, PyObject* kwargs)
{
	char *string;
	int insert_rank_node = 0;
	static char *kwlist[] = {"latex", "insert_rank_node", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|p", kwlist,
		&string, &insert_rank_node)) {
		PyErr_Format(PyExc_RuntimeError,
			"PyArg_ParseTupleAndKeywords error");
		return NULL;
	}

	PyObject *result = PyTuple_New(2);
	struct tex_parse_ret ret;
	ret = tex_parse(string);

	if (ret.code != PARSER_RETCODE_ERR) {
		struct optr_node* opt = ret.operator_tree;
		PyTuple_SetItem(result, 0, PyUnicode_FromString("OK"));

		if (opt) {
			PyObject *py_opt = NULL;
			//optr_print(opt, stdout); /* debug */
			py_opt = get_opt_pyobject(opt, insert_rank_node);
			PyTuple_SetItem(result, 1, py_opt);
			optr_release(opt);
		} else {
			Py_INCREF(Py_None);
			PyTuple_SetItem(result, 1, Py_None);
		}

		//subpaths_print(&ret.lrpaths, stdout); /* debug */
		subpaths_release(&ret.lrpaths);

	} else {
		PyTuple_SetItem(result, 0, PyUnicode_FromString(ret.msg));
		Py_INCREF(Py_None);
		PyTuple_SetItem(result, 1, Py_None);
	}

	return result;
}
