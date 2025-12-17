import os
import html
from bs4 import BeautifulSoup

# 读取 diff-cover 生成的 HTML 报告
def read_diffcover_html(html_file):
    with open(html_file, "r") as file:
        return file.read()

# 提取未覆盖行的行号
def extract_missing_lines(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    missing_lines = {}

    # 查找所有源文件的条目
    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) >= 3:
            file_name = cols[0].text.strip()
            missing_lines_str = cols[2].text.strip()
            # 如果有缺失的行号，进行处理
            if missing_lines_str:
                missing_lines[file_name] = parse_missing_lines(missing_lines_str)

    return missing_lines

# 解析缺失行号
def parse_missing_lines(missing_lines_str):
    lines = set()
    for part in missing_lines_str.split(","):
        if "-" in part:  # 处理区间
            start, end = map(int, part.split("-"))
            lines.update(range(start, end + 1))
        else:  # 处理单行
            lines.add(int(part))
    return lines

# 读取源代码并高亮未覆盖的行
def highlight_code(file_name, missing_lines):
    # 添加路径前缀
    file_path = f"infer_engines/vllm/{file_name}"
    
    try:
        with open(file_path, "r") as file:
            code = file.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

    highlighted_code = []
    for idx, line in enumerate(code, 1):
        line = line.rstrip()  # 保留换行符
        escaped_line = html.escape(line)  # 转义 HTML 特殊字符
        # 添加行号和代码
        if idx in missing_lines:
            # 高亮未覆盖的行，添加行号
            highlighted_code.append(f'<div style="background-color: #ffcccc; padding: 3px; font-family: monospace;">'
                                    f'<span style="color: #333;">{idx:4} </span>{escaped_line}</div>\n')
        else:
            # 正常显示已覆盖的行，添加行号
            highlighted_code.append(f'<div style="padding: 3px; font-family: monospace;">'
                                    f'<span style="color: #333;">{idx:4} </span>{escaped_line}</div>\n')

    return highlighted_code

# 生成最终的 HTML 报告
def generate_html_report(highlighted_code, file_name, output_dir):
    # 为每个文件生成一个独立的 HTML 文件
    # 确保目录存在
    output_html_file = os.path.join(output_dir, file_name + ".html")
    
    # 获取父目录并确保其存在
    os.makedirs(os.path.dirname(output_html_file), exist_ok=True)

    html_content = "<html><body><h1>Coverage Report</h1>"

    html_content += f"<h2>{file_name}</h2><pre style='white-space: pre-wrap; word-wrap: break-word; font-family: monospace;'>"
    for code in highlighted_code:
        html_content += code

    html_content += "</pre></body></html>"

    # 保存 HTML 文件
    with open(output_html_file, "w") as file:
        file.write(html_content)

def main():
    # 读取 diff-cover 生成的 HTML
    html_file = "coverage_html/patch_coverage.html"  # diff-cover 生成的报告文件
    html_content = read_diffcover_html(html_file)

    # 提取未覆盖的行号
    missing_lines = extract_missing_lines(html_content)

    # 输出文件夹
    output_dir = "coverage_html/highlighted_reports"
    os.makedirs(output_dir, exist_ok=True)

    # 遍历 vllm 目录下的所有源代码文件
    for file_name, lines in missing_lines.items():
        # 读取源代码并高亮显示未覆盖的行
        highlighted_code = highlight_code(file_name, lines)

        # 生成 HTML 报告并保存
        generate_html_report(highlighted_code, file_name, output_dir)

    print(f"HTML reports generated and saved in {output_dir}")

if __name__ == "__main__":
    main()