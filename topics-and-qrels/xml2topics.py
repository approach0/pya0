import os
import fire
from xmlr import xmliter


def xml2topics(input_xml='topics.arqmath-2022-task2-origin.xml',
    contextual=True):
    if not contextual:
        output_txt = input_xml.replace('xml', 'txt').replace('-origin', '')
        with open(output_txt, 'w') as fh:
            for attrs in xmliter(input_xml, 'Topic'):
                qid = attrs['@number']
                latex = attrs['Latex']
                print(f'{qid}\t{latex}', file=fh)
    else:
        output_txt = input_xml.replace('xml', 'json').replace('origin', 'context')
        import re
        import json
        from bs4 import BeautifulSoup
        def html2text(html, preserve):
            soup = BeautifulSoup(html, "html.parser")
            for elem in soup.select('span.math-container'):
                tex = elem.text.strip('$')
                if not preserve:
                    elem.replace_with('[imath]' + tex + '[/imath]')
                else:
                    formula_id = elem.get('id')
                    if formula_id is None:
                        elem.replace_with(' ')
                    else:
                        elem.replace_with(
                            f'[imath id="{formula_id}"]' + tex + '[/imath]'
                        )
            return soup.text
        output_topics = []
        for attrs in xmliter(input_xml, 'Topic'):
            qid = attrs['@number']
            formulaID = attrs['Formula_Id']
            title = attrs['Title']
            qbody = attrs['Question']
            post = html2text(title + '\n\n' + qbody, True)
            #print(qid, formulaID)
            #print(post, end="\n\n")
            #quit()
            matches = re.finditer(
                r'\[imath id="(q_\d+)"\](.*?)\[/imath\]', post
            )
            matches = list(reversed([m for m in matches]))
            for i, n in enumerate(matches):
                formulaID, tex = n.group(1), n.group(2)
                formula_len = len(tex.strip())
                if formulaID != formulaID:
                    continue
                ctx = post[:]
                for j, m in enumerate(matches):
                    span, _, tex = m.span(), m.group(1), m.group(2)
                    if i == j:
                        wrap = f'[imath]{tex}[/imath]'
                        ctx = ctx[:span[0]] + wrap + ctx[span[1]:]
                    else:
                        ctx = ctx[:span[0]] + '[MASK]' + ctx[span[1]:]
                output_topics.append({
                    'context': ctx,
                    'formulaID': formulaID,
                    'qid': qid
                })
                break

        with open(output_txt, 'w') as fh:
            json.dump(output_topics, fh, indent=4)


if __name__ == '__main__':
    os.environ["PAGER"] = 'cat'
    fire.Fire(xml2topics)
