import re

def replace_dollar_tex(s):
    l = len(s)
    i, j, stack = 0, 0, 0
    new_txt = ''
    while i < l:
        if s[i] == "\\" and (i + 1) < l:
            if s[i + 1] == '$':
                # skip if it is escaped dollar
                new_txt += '$'
                i += 1
            elif stack == 0:
                # otherwise just copy it
                new_txt += s[i]
        elif s[i] == '$':
            if stack == 0: # first open dollar
                stack = 1
                j = i + 1
            elif stack == 1: # second dollar
                if i == j:
                    # consecutive dollar
                    # (second open dollar)
                    stack = 2
                    j = i + 1
                else:
                    # non-consecutive dollar
                    # (close dollar)
                    stack = 0
                    # print('single: %s' % s[j:i])
                    new_txt += '[imath]%s[/imath]' % s[j:i]
            else: # stack == 2
                # first close dollar
                stack = 0
                # print('double: %s' % s[j:i])
                new_txt += '[imath]%s[/imath]' % s[j:i]
                # skip the second close dollar
                i += 1
        elif stack == 0:
            # non-escaped and non enclosed characters
            new_txt += s[i]
        i += 1
    return new_txt

def replace_display_tex(s):
    # replace '\[ * \]'
    regex = re.compile('\\\\\[(.+?)\\\\\]', re.DOTALL)
    return re.sub(regex, r"[imath]\1[/imath]", s)

def replace_inline_tex(s):
    # replace '\\( * \\)'
    regex = re.compile(r'\\\((.+)\\\)')
    return re.sub(regex, r"[imath]\1[/imath]", s)

# curl http://math.stackexchange.com/questions/1886701/justify-a-function-series-is-approximating-another-function
# to test everything.


def unwrap_tex_group(text, group_name):
    regex = re.compile(
            r"\\begin{" + group_name +
            r"\*?}(.+?)\\end{" + group_name +
            r"\*?}(?!\s*\[/imath\])", re.DOTALL) # negative lookahead
    return re.sub(regex, r"[imath]\1[/imath]", text)


def unwrap_tex_groups(text,
    groups=['align', 'alignat', 'equation', 'gather']):
    for grp in groups:
        text = unwrap_tex_group(text, grp)
    return text
