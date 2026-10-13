"""Build an offline static article and reports; no trackers or external scripts."""
import html
import re
import shutil
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'
REPO = 'https://github.com/lube8163-lab/semantic-image-communication-study'

CSS = '''
:root{color-scheme:light;--ink:#17283b;--muted:#55677c;--line:#d7e1e9;--accent:#16629a}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f7fa;color:var(--ink);font:17px/1.9 system-ui,-apple-system,"Segoe UI","Yu Gothic UI",sans-serif}
nav{padding:15px max(22px,calc((100vw - 1080px)/2));background:#14283d;color:white;display:flex;gap:24px;flex-wrap:wrap}nav a{color:#d6eaff;text-decoration:none;font-size:14px}
main{max-width:1060px;margin:28px auto;background:white;padding:44px 54px 64px;border:1px solid var(--line);border-radius:12px}h1{font-size:clamp(25px,3.7vw,37px);line-height:1.5;margin:12px 0 18px;letter-spacing:.01em}h2{font-size:25px;line-height:1.5;margin:52px 0 20px;padding-top:16px;border-top:1px solid var(--line)}h3{font-size:20px}
p{margin:20px 0}a{color:var(--accent);text-underline-offset:3px;overflow-wrap:anywhere}strong{font-weight:700}pre{overflow-x:auto;background:#f0f4f8;padding:20px;border-radius:8px;font-size:14px;line-height:1.6}code{font-size:.9em}img{max-width:100%;height:auto;display:block;margin:24px auto;border:1px solid var(--line);border-radius:5px}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.7;margin:24px 0}th,td{border:1px solid var(--line);padding:10px 11px;text-align:left}th{background:#eef4f8}tbody tr:nth-child(even){background:#fafcfe}.table-scroll{overflow:auto}.kicker{font-size:12px;letter-spacing:.1em;color:var(--muted)}.toc{font-size:14px;background:#f2f6fa;padding:18px 28px;border-radius:8px}.toc ul{padding-left:20px}.toc li{margin:3px 0}.toc a{text-decoration:none}.toc-title{font-weight:700}footer{color:var(--muted);font-size:13px;max-width:1060px;margin:28px auto;padding:0 20px 38px}.tag{display:inline-block;background:#e4f1eb;color:#22664e;padding:3px 10px;border-radius:99px;font-size:12px}.links{display:flex;gap:20px;flex-wrap:wrap}
@media(max-width:720px){body{font-size:16px}main{margin:0;border:0;border-radius:0;padding:26px 20px 44px}h2{font-size:22px}table{min-width:600px}nav{gap:16px}}
'''

def render(source, destination):
    text = source.read_text('utf-8')
    text = text.replace('](scripts/prepare_assets.py)', '](' + REPO + '/blob/main/scripts/prepare_assets.py)')
    if source.parent.name == 'articles':
        text = text.replace('../', '')
    text = re.sub(r'\]\(([^:)#]+)\.md(#[^)]*)?\)', lambda m: '](' + m[1] + '.html' + (m[2] or '') + ')', text)
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'], extension_configs={'toc': {'toc_depth': '2-3'}})
    body = md.convert(text)
    body = re.sub(r'<table>(.*?)</table>', r'<div class="table-scroll"><table>\1</table></div>', body, flags=re.S)
    body = re.sub(r'<img ', '<img loading="lazy" ', body)
    title = re.search(r'<h1[^>]*>(.*?)</h1>', body).group(1)
    relative = '../' if destination.parent.name == 'docs' else ''
    head = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Iris-3Bの中間特徴とVLMによる画像→文章→画像を、小規模な公開画像と実ペイロードで検証した結果。"><title>{html.unescape(title)}</title><link rel="stylesheet" href="{relative}style.css"></head><body><nav><a href="{relative}index.html">研究結果の記事</a><a href="{relative}results/text-route-20261013/report.html">全条件の画像比較</a><a href="{REPO}">GitHub / code</a><a href="{relative}ASSETS.html">画像出典・ライセンス</a></nav><main><div class="kicker">SEMANTIC IMAGE COMMUNICATION · LOCAL EXPLORATORY STUDY</div><span class="tag">実測結果 / 2026-10-14</span>'''
    if source.parent.name == 'articles':
        cut = body.find('<h2')
        body = body[:cut] + '<div class="toc"><div class="toc-title">この記事の内容</div>' + md.toc + '</div>' + body[cut:]
    footer = f'''</main><footer>lube8163-lab · 本文・独自の測定表はCC BY 4.0。外部写真は個別ライセンスを維持。再生成画像はAIの実験出力です。<p class="links"><a href="{relative}ASSETS.html">画像の作者と変更内容</a><a href="{REPO}/blob/main/NOTICE">NOTICE</a></p></footer></body></html>'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(head + body + footer, encoding='utf-8')

def main():
    SITE.mkdir(exist_ok=True)
    for name in ['results', 'data', 'sources', 'configs']:
        shutil.copytree(ROOT / name, SITE / name, dirs_exist_ok=True)
    shutil.copytree(ROOT / 'docs', SITE / 'docs', dirs_exist_ok=True)
    (SITE / 'style.css').write_text(CSS, encoding='utf-8')
    (SITE / '.nojekyll').write_text('', encoding='utf-8')
    render(ROOT / 'articles/semantic-image-communication-study.md', SITE / 'index.html')
    render(ROOT / 'ASSETS.md', SITE / 'ASSETS.html')
    for source in (ROOT / 'docs').glob('*.md'):
        render(source, SITE / 'docs' / (source.stem + '.html'))
    print('STATIC_SITE_READY', len(list(SITE.rglob('*'))), 'entries')

if __name__ == '__main__':
    main()
