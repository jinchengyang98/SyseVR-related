

def filter_line_numbers(text, special_marker='???'):# 这个函数的功能是为了过滤配对数据中的行号
    import re
    TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)

    # 使用正则表达式进行分词
    parts = TOKENIZER_RE.findall(text)
    filtered_parts = []

    for part in parts:
        # 检查是否包含特殊标记
        if special_marker in part:
            filtered_parts.append(part)
        else:
            # 如果部分不是仅由数字组成，则保留
            if not re.fullmatch(r'\d+', part):
                filtered_parts.append(part)

    # 重新组合处理过的部分
    filtered_text = ' '.join(filtered_parts)

    return filtered_text

def rm_line_numebr(data): # 这个函数的功能是为了去除数据中的行号
    # 使用正则表达式匹配每一行末尾的数字并移除
    # 表达式解释：\d+ 匹配一个或多个数字，$ 表示行末
    # re.MULTILINE 让 ^ 和 $ 匹配每一行的开头和结尾
    # 使用正则表达式匹配每一行末尾的数字并移除
    import re
    cleaned_lines = []
    for line in data.splitlines():
        cleaned_line = re.sub(r'\d+$', '', line)
        cleaned_lines.append(cleaned_line)
    # 返回处理后的数据，每行之间用空格连接
    return ' '.join(cleaned_lines)