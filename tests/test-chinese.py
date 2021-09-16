import os
import sys
import time
import json

sys.path.insert(0, '.')
import pya0

index_path = "tmp" # output index path
jieba_path = os.path.expanduser("~/cppjieba/dict")
highlight = True


if not os.path.exists(index_path):
    ix = pya0.index_open(index_path,
        segment_dict=jieba_path, highlight=highlight)
    if ix is None:
        print('Cannot open index.')
        quit(1)

    writer = pya0.index_writer(ix)

    pya0.writer_add_doc(
        writer,
        content="化简求值：若 [imath]4x^2+4x+y^2-6y+10=0[/imath]，试求[imath](x^2+2y^2+3)(x2+2y2-3)-(x-2y)^2(x+2y)^2[/imath]的值。",
        url="link1"
    )

    pya0.writer_add_doc(
        writer,
        content="已知函数[imath]f(x)=x^3+ax^2+bx+c[/imath]在[imath]x=-1[/imath]与[imath]x=2[/imath]处都取得极值。（1）求[imath]a[/imath]和[imath]b[/imath]的值及函数 [imath]f(x)[/imath]的单调区间；（2）若对[imath]x \in [-2，3][/imath]，不等式[imath]f(x)+c<c^2[/imath]恒成立，求[imath]c[/imath]的取值范围。",
        url="link2"
    )

    pya0.writer_maintain(writer, force=True)
    pya0.writer_close(writer)
    pya0.index_close(ix)


print('Searching...')
ix = pya0.index_open(index_path, option="r",
    segment_dict=jieba_path, highlight=highlight)
pya0.index_print_summary(ix)
JSON = pya0.search(ix, [
  {'str': 'x^3', 'type': 'tex'},
  {'str': '化简', 'type': 'term'},
], verbose=True)
J = json.loads(JSON)
if J['ret_code'] == 0:
    for i, hit in enumerate(J['hits']):
        print(i, hit['field_content'], end="\n\n")
else:
    print(J)
pya0.index_close(ix)
