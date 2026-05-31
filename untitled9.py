"""
栽培試験 GLM解析プラットフォーム (改善版+機能強化版)
主な機能追加:
  - ファイルアップロード機能 (CSV/Excel対応, Shift-JIS自動フォールバック)
  - 過分散 (Overdispersion) の自動計算と警告
  - Pandas 2.1.0 以降の警告対応 (.mapへの移行)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from matplotlib import font_manager
import io
import itertools
import warnings
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.drawing.image import Image as OpenpyxlImage

warnings.filterwarnings('ignore')

# ==========================================
# 定数・設定
# ==========================================
st.set_page_config(
    page_title="栽培試験 GLM解析プラットフォーム",
    page_icon="🌱",
    layout="wide"
)

ALPHA = 0.05  # 有意水準を一箇所で管理

# ==========================================
# 日本語フォント設定（安定化）
# ==========================================
@st.cache_resource
def setup_japanese_font():
    """日本語フォントを設定。成功したフォント名を返す。"""
    try:
        import japanize_matplotlib
        japanize_matplotlib.japanize()
        return "japanize_matplotlib"
    except ImportError:
        pass

    candidates = [
        'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP', 'Noto Sans JP',
        'Hiragino Sans', 'Hiragino Maru Gothic Pro',
        'MS Gothic', 'Yu Gothic', 'Meiryo', 'VL Gothic'
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            return font

    plt.rcParams['axes.unicode_minus'] = False
    return "fallback"

FONT_STATUS = setup_japanese_font()

# ==========================================
# ユーティリティ関数
# ==========================================

def reset_session():
    """セッション状態を完全リセット。"""
    keys_to_clear = [
        'analyzed', 'report_images', 'model_result',
        'report_glm', 'report_wald', 'df_eval',
        'eval_col', 'formula', 'target_col', 'dist_type', 'factor_cols',
        'factor_types', 'dispersion' # ★追加: 過分散パラメータもクリア
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

def get_cld_letters(groups_sorted: list, tukey_df: pd.DataFrame) -> dict:
    """Tukey検定結果から compact letter display (CLD) を生成。"""
    required_cols = {'group1', 'group2', 'reject'}
    if not required_cols.issubset(tukey_df.columns):
        cols = tukey_df.columns.tolist()
        if len(cols) >= 3:
            tukey_df = tukey_df.copy()
            tukey_df.columns = ['group1', 'group2'] + cols[2:]
        else:
            return {g: 'a' for g in groups_sorted}

    ns_pairs = set()
    for _, row in tukey_df.iterrows():
        g1, g2 = str(row['group1']), str(row['group2'])
        if not row['reject']:
            ns_pairs.add((g1, g2))
            ns_pairs.add((g2, g1))
    for g in groups_sorted:
        ns_pairs.add((g, g))

    letters = {g: [] for g in groups_sorted}
    letter_groups = []
    current_idx = 0

    for gi in groups_sorted:
        absorb = [gj for gj in groups_sorted if (gi, gj) in ns_pairs]
        matched = any(set(lg) == set(absorb) for lg in letter_groups)
        if not matched:
            new_letter = chr(ord('a') + current_idx)
            current_idx += 1
            letter_groups.append(absorb)
            for g in absorb:
                letters[g].append(new_letter)

    return {g: ''.join(sorted(set(v))) for g, v in letters.items()}

def detect_perfect_separation(df: pd.DataFrame, target_col: str, factor_col: str) -> list:
    """カテゴリ変数における完全分離の事前検出。"""
    problems = []
    unique_vals = df[target_col].dropna().unique()
    is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})

    if is_binary:
        for cat, grp in df.groupby(factor_col):
            vals = grp[target_col].dropna()
            if len(vals) > 0 and vals.nunique() == 1:
                problems.append(str(cat))
    return problems

def safe_numeric(series: pd.Series) -> pd.Series:
    """安全に数値変換。変換不可は NaN。"""
    return pd.to_numeric(series, errors='coerce')

def make_fig_for_category(df, factor, eval_col, target_col, groups_sorted, final_report):
    """カテゴリ変数のBoxplot + CLD を生成。"""
    n_groups = len(groups_sorted)
    fig_w = max(5, 3 + n_groups * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    sns.boxplot(
        x=factor, y=eval_col, data=df,
        order=groups_sorted, ax=ax,
        color='#f0f0f0', showfliers=False,
        linewidth=1.2
    )
    sns.stripplot(
        x=factor, y=eval_col, data=df,
        order=groups_sorted, ax=ax,
        color='black', alpha=0.55, size=4, jitter=True
    )

    for gname in groups_sorted:
        rows = final_report[final_report[factor].astype(str) == str(gname)]
        if rows.empty:
            continue
        letter = rows['cld'].values[0]
        xpos = groups_sorted.index(gname)
        target_data = df[df[factor].astype(str) == str(gname)][eval_col]
        if target_data.empty:
            continue
        ymax = target_data.max()
        yrange = df[eval_col].max() - df[eval_col].min()
        offset = yrange * 0.05 if yrange > 0 else 0.02
        ax.text(
            xpos, ymax + offset, letter,
            ha='center', va='bottom', fontweight='bold',
            color='crimson', fontsize=11
        )

    ax.set_xlabel(factor, fontsize=10)
    ax.set_ylabel(target_col, fontsize=10)
    ax.tick_params(axis='x', rotation=30 if n_groups > 5 else 0)
    plt.tight_layout()
    return fig

def make_fig_for_numeric(df, factor, eval_col, target_col, plot_logistic):
    """数値変数の回帰プロットを生成。"""
    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        sns.regplot(
            x=factor, y=eval_col, data=df, ax=ax,
            scatter_kws={'alpha': 0.55, 'color': '#333333', 's': 30},
            line_kws={'color': 'crimson', 'linewidth': 1.5},
            logistic=plot_logistic
        )
    except Exception:
        sns.regplot(
            x=factor, y=eval_col, data=df, ax=ax,
            scatter_kws={'alpha': 0.55, 'color': '#333333', 's': 30},
            line_kws={'color': 'crimson', 'linewidth': 1.5},
            logistic=False
        )
    ax.set_xlabel(factor, fontsize=10)
    ax.set_ylabel(target_col, fontsize=10)
    plt.tight_layout()
    return fig

def fig_to_bytesio(fig) -> io.BytesIO:
    """MatplotlibのFigureをBytesIOに変換。"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_ai_prompt(target_col, dist_type, formula, report_glm, report_wald):
    """LLM向けのMarkdownプロンプトを生成。"""
    try:
        glm_md = report_glm.reset_index().to_markdown(index=False, floatfmt='.4f')
    except Exception:
        glm_md = report_glm.reset_index().to_csv(sep='\t', index=False)

    if report_wald is not None:
        try:
            wald_md = report_wald.reset_index().to_markdown(index=False, floatfmt='.4f')
        except Exception:
            wald_md = report_wald.reset_index().to_csv(sep='\t', index=False)
    else:
        wald_md = "計算不可（データなし）"

    clean_formula = formula.replace('Q(', '').replace(')', '').replace('"', '')

    prompt = f"""あなたは農業統計と栽培技術の専門家です。以下の栽培試験データにおける一般化線形モデル(GLM)の解析結果を解釈し、実践的なアドバイスを提供してください。

## 解析の前提条件
- **目的変数**: {target_col}
- **確率分布**: {dist_type}
- **モデル式**: `{clean_formula}`
- **有意水準**: α = {ALPHA}

## Wald検定表（要因全体の有意性）
{wald_md}

## 回帰係数（個別水準の効果量）
{glm_md}

## 解釈の指示
1. **Wald検定表**から、有意な要因（Pr < {ALPHA}）とそうでない要因を区別して説明してください。
2. **回帰係数の Estimate** の正負・大きさから、水準間の実務的な違いを具体的に説明してください。
3. **Std.Error が異常に大きい** 項目があれば、完全分離・交絡・データ不足を疑い、その理由と対処法を述べてください。
4. この結果を踏まえ、**現場の栽培管理または次回の試験設計**にどう活かすべきか提案してください。
"""
    return prompt

def generate_excel_report(target_col, dist_type, formula, report_glm, report_wald, report_images: dict) -> bytes:
    """A4横幅対応のExcelレポートを生成。"""
    output = io.BytesIO()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "解析レポート"

    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToPage = True
    ws.page_setup.fitToHeight = False
    ws.page_setup.fitToWidth = 1

    title_font   = Font(size=14, bold=True)
    head_font    = Font(bold=True, color="FFFFFF")
    head_fill    = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
    subhead_font = Font(bold=True, size=11)
    thin_bd = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
    sig_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    def write_header_row(ws, row, headers, col_start=1):
        for i, h in enumerate(headers):
            c = ws.cell(row=row, column=col_start + i, value=h)
            c.font = head_font; c.fill = head_fill; c.border = thin_bd
            c.alignment = Alignment(horizontal='center')

    def write_data_cell(ws, row, col, value, bold=False, highlight=False):
        c = ws.cell(row=row, column=col, value=value)
        c.border = thin_bd
        if bold: c.font = Font(bold=True)
        if highlight: c.fill = sig_fill
        return c

    cur = 1

    ws.cell(row=cur, column=1, value="🌱 栽培試験 GLM解析レポート").font = title_font
    cur += 2
    ws.cell(row=cur, column=1, value="【モデル設定】").font = subhead_font
    cur += 1
    ws.cell(row=cur, column=1, value=f"目的変数: {target_col}")
    ws.cell(row=cur, column=3, value=f"確率分布: {dist_type}")
    cur += 1
    clean_f = formula.replace('Q(', '').replace(')', '').replace('"', '')
    ws.cell(row=cur, column=1, value=f"モデル式: {clean_f}")
    ws.cell(row=cur, column=4, value=f"有意水準: α = {ALPHA}")
    cur += 2

    ws.cell(row=cur, column=1, value="【1. Wald検定表 (要因の有意性)】").font = subhead_font
    cur += 1
    if report_wald is not None:
        headers_w = ["要因", "Df", "Chi-Square", "Pr(>Chisq)", "有意"]
        write_header_row(ws, cur, headers_w)
        cur += 1
        for idx, row_data in report_wald.iterrows():
            is_sig = float(row_data['Pr(>Chisq)']) < ALPHA
            write_data_cell(ws, cur, 1, str(idx), highlight=is_sig)
            write_data_cell(ws, cur, 2, int(row_data['Df']) if pd.notna(row_data['Df']) else '', highlight=is_sig)
            write_data_cell(ws, cur, 3, round(float(row_data['Chi-Square']), 3) if pd.notna(row_data['Chi-Square']) else '', highlight=is_sig)
            write_data_cell(ws, cur, 4, round(float(row_data['Pr(>Chisq)']), 4) if pd.notna(row_data['Pr(>Chisq)']) else '', highlight=is_sig)
            write_data_cell(ws, cur, 5, row_data.get('Signif', ''), highlight=is_sig)
            cur += 1
    else:
        ws.cell(row=cur, column=1, value="（計算失敗）")
        cur += 1
    cur += 2

    ws.cell(row=cur, column=1, value="【2. 回帰係数 (Summary)】").font = subhead_font
    cur += 1
    headers_g = ["要因 / 水準", "Estimate", "Std.Error", "z value", "Pr(>|z|)", "有意"]
    write_header_row(ws, cur, headers_g)
    cur += 1
    for str_idx, row_data in report_glm.iterrows():
        is_sig = float(row_data['Pr(>|z|)']) < ALPHA if pd.notna(row_data['Pr(>|z|)']) else False
        write_data_cell(ws, cur, 1, str(str_idx), highlight=is_sig)
        for ci, col_name in enumerate(['Estimate', 'Std.Error', 'z value', 'Pr(>|z|)'], 2):
            val = row_data[col_name]
            write_data_cell(ws, cur, ci, round(float(val), 4) if pd.notna(val) else '', highlight=is_sig)
        write_data_cell(ws, cur, 6, row_data.get('Signif', '') if not pd.isna(row_data['Std.Error']) and row_data['Std.Error'] <= 10 else '⚠️SE過大', highlight=is_sig)
        cur += 1
    cur += 3

    if report_images:
        ws.cell(row=cur, column=1, value="【3. 要因別の影響グラフ】").font = subhead_font
        cur += 2
        for factor_name, img_buf in report_images.items():
            ws.cell(row=cur, column=1, value=f"▶ {factor_name}").font = Font(bold=True)
            cur += 1
            img_buf.seek(0)
            try:
                xl_img = OpenpyxlImage(img_buf)
                xl_img.width  = int(xl_img.width  * 0.75)
                xl_img.height = int(xl_img.height * 0.75)
                ws.add_image(xl_img, f"B{cur}")
                rows_to_skip = max(int(xl_img.height / 18) + 2, 5)
            except Exception:
                ws.cell(row=cur, column=1, value="（グラフ挿入失敗）")
                rows_to_skip = 2
            cur += rows_to_skip

    ws.column_dimensions['A'].width = 35
    for col_letter in ['B', 'C', 'D', 'E', 'F']: ws.column_dimensions[col_letter].width = 14

    wb.save(output)
    return output.getvalue()


# ==========================================
# アプリ本体
# ==========================================
st.title("📈 栽培試験データ GLM解析アプリ")
st.markdown("確率分布・質的/量的変数の組み合わせに対応した **一般化線形モデル (GLM)** による解析プラットフォームです。")

if FONT_STATUS == "fallback":
    st.warning("⚠️ 日本語フォントが見つかりませんでした。文字化けする場合は `pip install japanize-matplotlib` を実行してください。")

with st.sidebar:
    st.header("⚙️ 操作")
    if st.button("🔄 リセット（最初からやり直す）", use_container_width=True):
        reset_session()
    st.divider()
    st.caption(f"有意水準: α = {ALPHA}")
    st.caption("GLM engine: statsmodels")

# ==========================================
# Step 1: データ読み込み (★修正: ファイルアップロード対応)
# ==========================================
st.header("1. データの読み込み")
data_source = st.radio(
    "入力方法を選択：",
    ["📄 ファイルアップロード", "📋 Excelデータを貼り付け", "🥔 サンプルデータで試す"],
    horizontal=True
)
df_raw = None

if data_source == "📄 ファイルアップロード":
    # ★追加: st.file_uploaderの利用
    uploaded_file = st.file_uploader("CSV または Excelファイルをアップロードしてください", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_raw = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    # Shift-JISでの読み込みを試行 (Excel出力のCSV対策)
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, encoding='shift_jis')
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.success(f"✅ ファイルを読み込みました（{len(df_raw)} 件）。")
        except Exception as e:
            st.error(f"❌ 読み込みエラー: {e}")

elif data_source == "🥔 サンプルデータで試す":
    rng = np.random.default_rng(42)
    rows = []
    for rep, v, w in itertools.product(range(1, 6), ['シマアカリ', 'ニシユタカ', 'アイユタカ'], [10, 20, 30]):
        base_p = 0.5
        if v == 'シマアカリ': base_p += 0.2
        if w == 30: base_p += 0.15
        elif w == 10: base_p -= 0.2
        prob = float(np.clip(rng.normal(base_p, 0.1), 0.05, 0.95))
        total = 20
        sprouted = int(rng.binomial(total, prob))
        rows.append([rep, v, w, total, sprouted, sprouted / total])
    df_raw = pd.DataFrame(rows, columns=['rep', 'var', 'water_ml', 'total', 'sprouted', 'rate'])
    st.success("✅ サンプルデータを読み込みました。")

else:
    col_paste1, col_paste2 = st.columns([3, 1])
    with col_paste2:
        sep_choice = st.selectbox("区切り文字", ["自動検出", "タブ (\\t)", "カンマ (,)", "スペース"])
    with col_paste1:
        pasted_data = st.text_area("Excel/CSVデータを貼り付け (Ctrl+V)", height=100)

    if pasted_data:
        sep_map = {"自動検出": None, "タブ (\\t)": '\t', "カンマ (,)": ',', "スペース": r'\s+'}
        sep_val = sep_map[sep_choice]
        try:
            if sep_val is None:
                df_raw = pd.read_csv(io.StringIO(pasted_data), sep=None, engine='python')
            elif sep_val == r'\s+':
                df_raw = pd.read_csv(io.StringIO(pasted_data), sep=r'\s+', engine='python')
            else:
                df_raw = pd.read_csv(io.StringIO(pasted_data), sep=sep_val)
            st.success(f"✅ データを読み込みました（{len(df_raw)} 件）。")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

if df_raw is None:
    st.stop()

# ==========================================
# Step 2: データの絞り込み
# ==========================================
st.header("2. データの絞り込み（層別解析）")
df_target = df_raw.copy()
filter_cols = st.multiselect("絞り込みに使用する列を選択（複数可）:", df_raw.columns.tolist())
if filter_cols:
    filter_columns = st.columns(min(len(filter_cols), 3))
    for i, col in enumerate(filter_cols):
        with filter_columns[i % 3]:
            unique_vals = df_raw[col].dropna().unique().tolist()
            selected_vals = st.multiselect(f"【{col}】", unique_vals, default=unique_vals, key=f"filter_{col}")
            df_target = df_target[df_target[col].isin(selected_vals)]

st.info(f"📊 解析対象: **{len(df_target)} 件** （元データ: {len(df_raw)} 件）")
if df_target.empty:
    st.error("⚠️ 絞り込みの結果がゼロ件です。")
    st.stop()

st.divider()

# ==========================================
# Step 3: データの要約と交絡チェック
# ==========================================
st.header("3. データの要約と交絡チェック")
col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown("**データプレビュー**")
    st.dataframe(df_target.head(5), use_container_width=True)
with col_s2:
    st.markdown("**要約統計量**")
    st.dataframe(df_target.describe(), use_container_width=True)

st.subheader("📊 交絡チェック（クロス集計）")
cols = df_target.columns.tolist()
cat_cols = [c for c in cols if df_target[c].dtype == object or df_target[c].nunique() < 20]

col_c1, col_c2 = st.columns(2)
with col_c1: cross_var1 = st.selectbox("行変数", ["---"] + cat_cols)
with col_c2: cross_var2 = st.selectbox("列変数", ["---"] + cat_cols)

if cross_var1 != "---" and cross_var2 != "---" and cross_var1 != cross_var2:
    try:
        ct = pd.crosstab(df_target[cross_var1], df_target[cross_var2])
        st.dataframe(ct, use_container_width=True)
        if (ct == 0).sum().sum() > 0:
            st.warning("⚠️ ゼロセルが存在します。完全分離エラーに注意してください。")
    except Exception as e:
        st.error(f"クロス集計エラー: {e}")

st.divider()

# ==========================================
# Step 4: モデル設定
# ==========================================
st.header("4. モデル設定")
col_m1, col_m2 = st.columns(2)
numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_target[c])]

with col_m1:
    target_col = st.selectbox("目的変数（応答変数）", numeric_cols)
    dist_type = st.selectbox("確率分布 (family)", [
        "二項分布 (Binomial / 発生数・0/1)",
        "ポアソン分布 (Poisson / カウントデータ)",
        "正規分布 (Gaussian / 連続値)"
    ])
    b_fmt, total_col = None, None
    if "二項分布" in dist_type:
        b_fmt = st.radio("データ形式", ["発生数 と 調査数 (集計データ)", "0, 1 データ (個体ごと)", "割合 (0〜1)"], horizontal=True)
        if b_fmt == "発生数 と 調査数 (集計データ)":
            total_col = st.selectbox("「調査数 (母数)」の列", [c for c in numeric_cols if c != target_col])

with col_m2:
    all_factor_candidates = [c for c in cols if c != target_col]
    safe_defaults = [c for c in all_factor_candidates if c.lower() not in ('rep', '反復', '繰り返し') and df_target[c].nunique() < len(df_target) * 0.9][:3]
    factor_cols = st.multiselect("要因（説明変数）", all_factor_candidates, default=safe_defaults)

factor_types = {}
if factor_cols:
    st.markdown("**各要因のデータ型：**")
    for f in factor_cols:
        is_text = df_target[f].dtype == object
        default_type = "カテゴリ (質的変数)" if is_text or df_target[f].nunique() <= 10 else "数値 (連続量)"
        factor_types[f] = st.radio(f"【{f}】", ["カテゴリ (質的変数)", "数値 (連続量)"], index=0 if "カテゴリ" in default_type else 1, horizontal=True, key=f"type_{f}")

run_button = st.button("🚀 GLMを実行", type="primary", disabled=(len(factor_cols) == 0))

# ==========================================
# Step 5: GLM実行
# ==========================================
if run_button and factor_cols and target_col:
    for k in ['report_images', 'model_result', 'report_glm', 'report_wald', 'df_eval', 'eval_col', 'formula', 'ai_prompt', 'excel_data', 'dispersion']:
        if k in st.session_state: del st.session_state[k]

    st.divider()
    st.header("5. 一般化線形モデルのあてはめ結果")

    df_eval = df_target.copy()
    df_eval[target_col] = safe_numeric(df_eval[target_col])
    df_eval = df_eval.dropna(subset=[target_col] + factor_cols)

    if df_eval.empty: st.error("⚠️ データが0件になりました。"); st.stop()

    eval_col = f"{target_col}__ModelVal"
    model_weights = None

    if "二項分布" in dist_type:
        family = sm.families.Binomial()
        if b_fmt == "0, 1 データ (個体ごと)":
            df_eval[eval_col] = np.clip(df_eval[target_col], 0, 1)
        elif b_fmt == "発生数 と 調査数 (集計データ)":
            df_eval = df_eval.dropna(subset=[total_col])
            df_eval[total_col] = safe_numeric(df_eval[total_col])
            df_eval = df_eval[df_eval[total_col] > 0]
            if df_eval.empty: st.error("⚠️ 調査数が0または欠損のためデータが0件になりました。"); st.stop()
            df_eval[eval_col] = df_eval[target_col] / df_eval[total_col]
            model_weights = df_eval[total_col]
        elif b_fmt == "割合 (0〜1)":
            max_val = df_eval[target_col].max() if not df_eval.empty else 0
            df_eval[eval_col] = np.clip(df_eval[target_col] / 100.0 if max_val > 1.5 else df_eval[target_col], 0.0, 1.0)
    elif "ポアソン分布" in dist_type:
        family = sm.families.Poisson()
        df_eval[eval_col] = df_eval[target_col].round().astype(int)
    else:
        family = sm.families.Gaussian()
        df_eval[eval_col] = df_eval[target_col]

    sep_warnings = []
    for f in factor_cols:
        if "カテゴリ" in factor_types[f]:
            prob_cats = detect_perfect_separation(df_eval, eval_col, f)
            if prob_cats:
                sep_warnings.append(f"**{f}**: カテゴリ {', '.join(prob_cats)} で応答値が均一 → 完全分離の可能性")
    if sep_warnings:
        with st.expander("⚠️ 完全分離の可能性（事前チェック）", expanded=True):
            for w in sep_warnings: st.warning(w)

    formula_terms = []
    for f in factor_cols:
        if "カテゴリ" in factor_types[f]:
            df_eval[f] = df_eval[f].astype(str)
            formula_terms.append(f'C(Q("{f}"))')
        else:
            df_eval[f] = safe_numeric(df_eval[f])
            formula_terms.append(f'Q("{f}")')

    df_eval = df_eval.dropna(subset=factor_cols)
    formula = f'Q("{eval_col}") ~ ' + ' + '.join(formula_terms)

    try:
        with st.spinner("GLMを計算中..."):
            try:
                if model_weights is not None:
                    model = smf.glm(formula, data=df_eval, family=family, var_weights=model_weights).fit()
                else:
                    model = smf.glm(formula, data=df_eval, family=family).fit()
            except PerfectSeparationError:
                st.error("❌ **完全分離エラー**\n\n特定のカテゴリで応答が完全に0または1に偏っています。")
                st.stop()
            except Exception as fit_e:
                st.error(f"❌ モデル計算エラー: {fit_e}")
                st.stop()

        if model.df_resid <= 0:
            st.error("❌ 残差自由度が0以下です。パラメータ数がデータ数を超えています。")
            st.stop()

        # ★追加: 過分散パラメータ(Overdispersion)の計算
        dispersion = None
        if "ポアソン分布" in dist_type or ("二項分布" in dist_type and b_fmt == "発生数 と 調査数 (集計データ)"):
            if model.df_resid > 0:
                dispersion = model.pearson_chi2 / model.df_resid

        report_glm = pd.DataFrame({
            'Estimate':  model.params,
            'Std.Error': model.bse,
            'z value':   model.tvalues,
            'Pr(>|z|)':  model.pvalues
        })
        report_glm['Signif'] = report_glm['Pr(>|z|)'].apply(lambda p: "**" if p < 0.01 else ("*" if p < ALPHA else "n.s."))
        report_glm['Note'] = report_glm['Std.Error'].apply(lambda se: "⚠️SE過大" if pd.notna(se) and se > 10 else "")

        report_wald = None
        try:
            wald_res = model.wald_test_terms().table
            report_wald = wald_res[['df_constraint', 'statistic', 'pvalue']].copy()
            report_wald.columns = ['Df', 'Chi-Square', 'Pr(>Chisq)']
            for col_w in report_wald.columns:
                report_wald[col_w] = pd.to_numeric(report_wald[col_w], errors='coerce').astype(float)
            report_wald['Signif'] = report_wald['Pr(>Chisq)'].apply(lambda p: "**" if p < 0.01 else ("*" if p < ALPHA else "n.s."))
        except Exception:
            pass

        st.session_state['report_glm']  = report_glm
        st.session_state['report_wald'] = report_wald
        st.session_state['df_eval']     = df_eval
        st.session_state['eval_col']    = eval_col
        st.session_state['formula']     = formula
        st.session_state['target_col']  = target_col
        st.session_state['dist_type']   = dist_type
        st.session_state['factor_cols'] = factor_cols
        st.session_state['factor_types']= factor_types
        st.session_state['dispersion']  = dispersion # ★追加
        st.session_state['analyzed']    = True

    except Exception as e:
        st.error(f"❌ 予期せぬエラー: {e}")
        st.stop()


# ==========================================
# Step 6以降: セッション状態から結果を表示
# ==========================================
if st.session_state.get('analyzed'):

    report_glm   = st.session_state['report_glm']
    report_wald  = st.session_state['report_wald']
    df_eval      = st.session_state['df_eval']
    eval_col     = st.session_state['eval_col']
    formula      = st.session_state['formula']
    target_col   = st.session_state['target_col']
    dist_type    = st.session_state['dist_type']
    factor_cols  = st.session_state['factor_cols']
    factor_types = st.session_state['factor_types']
    dispersion   = st.session_state.get('dispersion') # ★追加

    tab_res, tab_ai = st.tabs(["📊 解析サマリー", "🤖 AI解析用プロンプト生成"])

    with tab_res:
        st.code(f"モデル式: {formula.replace('Q(', '').replace(')', '').replace(chr(34), '')}", language="r")
        
        # ★追加: 過分散の警告表示
        if dispersion is not None:
            st.markdown(f"**💡 過分散パラメータ (Pearson χ² / df): {dispersion:.2f}**")
            if dispersion > 1.5:
                st.warning(
                    "⚠️ **過分散の可能性があります。** データのばらつきが想定より大きくなっています。\n"
                    "そのまま解釈すると「有意差が出やすくなる（第1種の過誤）」リスクがあるため、"
                    "解釈に注意するか、別の分布（負の二項分布など）への変更を検討してください。"
                )
            elif dispersion < 0.5:
                st.info("ℹ️ 過小分散の傾向があります。")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**■ 回帰係数 (summary)**")
            
            # ★修正: Pandas >= 2.1.0 対応のため applymap -> map へ変更
            # 念のため古いバージョンも考慮して try-except を記述
            styled_df = report_glm.style.format({
                'Estimate': '{:.4f}', 'Std.Error': '{:.4f}',
                'z value': '{:.3f}', 'Pr(>|z|)': '{:.4f}'
            })
            
            try:
                styled_df = styled_df.map(
                    lambda v: 'background-color: #fff3cd' if '⚠️' in str(v) else '',
                    subset=['Note']
                )
            except AttributeError:
                styled_df = styled_df.applymap(
                    lambda v: 'background-color: #fff3cd' if '⚠️' in str(v) else '',
                    subset=['Note']
                )
                
            st.dataframe(styled_df, use_container_width=True)

        with col_r2:
            st.markdown("**■ Wald検定表 (anova Chisq)**")
            if report_wald is not None:
                st.dataframe(
                    report_wald.style.format({
                        'Df': '{:.0f}', 'Chi-Square': '{:.2f}', 'Pr(>Chisq)': '{:.4f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("Wald検定の計算に失敗しました。")

    with tab_ai:
        st.markdown("### AIへの解析指示プロンプト")
        st.caption("このテキストを ChatGPT / Gemini / Claude などに貼り付けてください。")
        ai_prompt = generate_ai_prompt(target_col, dist_type, formula, report_glm, report_wald)
        
        # ★追加: 過分散情報をプロンプトにも補足
        if dispersion is not None and dispersion > 1.5:
            ai_prompt += f"\n5. **過分散 (Overdispersion) について**: 過分散パラメータが {dispersion:.2f} と高くなっています。この点に関する統計的注意点も付記してください。\n"
            
        st.text_area("📋 プロンプトをコピー", value=ai_prompt, height=420)

    st.divider()

    st.header("6. 各要因の影響と可視化")
    if 'report_images' not in st.session_state:
        st.session_state['report_images'] = {}

    for factor in factor_cols:
        if factor.lower() in ('rep', '反復', '繰り返し', 'block', 'blk'):
            continue

        is_cat = "カテゴリ" in factor_types[factor]
        st.subheader(f"▶ {factor}  ({'質的変数' if is_cat else '量的変数'})")

        if is_cat:
            summary_stats = (
                df_eval.groupby(factor)[eval_col]
                .agg(['count', 'mean', 'std'])
                .reset_index()
                .rename(columns={'count': 'N', 'mean': 'Mean (解析値)', 'std': 'SD'})
            )
            
            # ★追加: Tukeyの計算前エラー回避ロジック
            unique_groups = df_eval[factor].nunique()
            if unique_groups < 2:
                st.warning(f"「{factor}」は水準が1つしかないため、多重比較をスキップします。")
                st.dataframe(summary_stats.style.format(precision=3), use_container_width=True)
                continue

            try:
                tukey_obj = pairwise_tukeyhsd(endog=df_eval[eval_col], groups=df_eval[factor].astype(str), alpha=ALPHA)
                tukey_df = pd.DataFrame(data=tukey_obj._results_table.data[1:], columns=tukey_obj._results_table.data[0])
                tukey_df.columns = [str(c) for c in tukey_df.columns]

                means_sorted = df_eval.groupby(factor)[eval_col].mean().sort_values(ascending=False)
                groups_sorted = means_sorted.index.astype(str).tolist()
                cld_map = get_cld_letters(groups_sorted, tukey_df)

                letters_df = pd.DataFrame({factor: list(cld_map.keys()), 'cld': list(cld_map.values())})
                summary_stats[factor] = summary_stats[factor].astype(str)
                final_report = pd.merge(summary_stats, letters_df, on=factor).sort_values('Mean (解析値)', ascending=False).reset_index(drop=True)

                col_tk1, col_tk2 = st.columns([2, 3])
                with col_tk1:
                    st.markdown(f"**Tukeyの多重比較 (α={ALPHA})**")
                    st.dataframe(final_report.style.format(precision=3), use_container_width=True)
                with col_tk2:
                    fig = make_fig_for_category(df_eval, factor, eval_col, target_col, groups_sorted, final_report)
                    st.pyplot(fig)
                    st.session_state['report_images'][factor] = fig_to_bytesio(fig)

            except Exception as e:
                st.warning(f"多重比較の計算ができませんでした: {e}")
                st.dataframe(summary_stats.style.format(precision=3), use_container_width=True)

        else:
            col_n1, col_n2 = st.columns([2, 3])
            with col_n1:
                st.markdown("**変数の要約**")
                st.dataframe(df_eval[factor].describe(), use_container_width=True)
            with col_n2:
                plot_logistic = False
                if "二項分布" in dist_type:
                    unique_vals_set = set(df_eval[eval_col].dropna().unique())
                    if unique_vals_set.issubset({0, 1, 0.0, 1.0}):
                        plot_logistic = True

                fig = make_fig_for_numeric(df_eval, factor, eval_col, target_col, plot_logistic)
                st.pyplot(fig)
                st.session_state['report_images'][factor] = fig_to_bytesio(fig)

    st.divider()

    st.header("7. 解析レポートのダウンロード (A4印刷対応)")
    if 'excel_data' not in st.session_state:
        with st.spinner("Excelレポートを生成中..."):
            try:
                st.session_state['excel_data'] = generate_excel_report(
                    target_col, dist_type, formula, report_glm, report_wald, st.session_state.get('report_images', {})
                )
            except Exception as e:
                st.error(f"Excelレポート生成エラー: {e}")
                st.session_state['excel_data'] = None

    if st.session_state.get('excel_data'):
        st.download_button(
            label="📥 解析レポート（図入りExcel）をダウンロード",
            data=st.session_state['excel_data'],
            file_name="GLM_Analysis_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    else:
        st.info("Excelレポートの生成に失敗しました。解析を再実行してください。")