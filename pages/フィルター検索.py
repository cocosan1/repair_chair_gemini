import streamlit as st
import pandas as pd
import os
from PIL import Image
import pathlib

# --- ページ設定を最初に行う ---
st.set_page_config(layout="wide")

# --- 画像とキャプションのスタイル調整 ---
# (スタイル部分は変更なし)
style = """
<style>
/* 画像要素自体のスタイル */
div[data-testid="stImage"] img {
    max-height: 200px;  /* ★ 画像の最大高さを指定 (この値を調整) */
    object-fit: contain; /* アスペクト比を保ちコンテナ内に収める */
    display: block;      /* 中央揃えのためにブロック要素に */
    margin-left: auto;   /* 画像を中央揃え */
    margin-right: auto;  /* 画像を中央揃え */
    width: auto;         /* 高さに合わせて幅を自動調整 */
}
/* キャプションのスタイル */
figcaption {
    font-size: 18px !important; /* キャプションの文字サイズ */
    color: #555;             /* キャプションの色 */
    text-align: center;      /* キャプションを中央揃え */
}
</style>
"""
st.markdown(style, unsafe_allow_html=True)
# --- スタイル調整ここまで ---


# --- 定数 ---
EXCEL_FILE = "chair_classification_results.xlsx"
DETAILS_EXCEL_FILE = "chair_all.xlsx"
PRICE_LIST_EXCEL_PATH = "修理価格一覧app用.xlsx"
COMMENT_EXCEL_FILE = "comment.xlsx"
NOTES_EXCEL_PATH = "修理注意点.xlsx"
IMAGE_FOLDER = "images"
COLUMNS_TO_FILTER = ["背のデザイン", "座面", "肘"]
FILENAME_COLUMN = "ファイル名" # chair_classification_results.xlsx のファイル名列
DETAILS_HINBAN_COLUMN = "品番" # chair_all.xlsx の品番列名
COMMENT_HINBAN_COLUMN = "品番" # comment.xlsx の品番列名
COLOR_FLAG_COLUMN = "赤/黄/白" # ★ 追加: chair_classification_results.xlsx のフラグ列
NUM_DISPLAY_COLUMNS = 6

# --- データ読み込み関数 ---
@st.cache_data
def load_classification_data(file_path, filename_col, cols_to_clean, flag_col):
    """分類情報Excelファイルを読み込み、指定列をクリーニングしてDataFrameを返す"""
    try:
        df = pd.read_excel(file_path)
        if filename_col not in df.columns:
            st.warning(f"警告: ファイル '{file_path}' に '{filename_col}' 列が見つかりません。")
            return None
        df[filename_col] = df[filename_col].fillna('').astype(str).str.strip()

        # フラグ列の存在チェックと型変換（数値として扱う）
        if flag_col in df.columns:
            # 数値に変換できない値は NaN になり、fillna(0) で 0 にする
            df[flag_col] = pd.to_numeric(df[flag_col], errors='coerce').fillna(0).astype(int) # 整数型に変換
        else:
            st.warning(f"警告: ファイル '{file_path}' に '{flag_col}' 列が見つかりません。「更に見る」機能は利用できません。")
            # 列が存在しない場合は、便宜上 0 で埋めた列を追加しておく
            df[flag_col] = 0

        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.strip().str.replace('　', ' ', regex=False).str.strip()
        return df
    except FileNotFoundError:
        st.error(f"エラー: ファイル '{file_path}' が見つかりません。")
        st.info(f"スクリプトと同じディレクトリに '{file_path}' ファイルを配置してください。")
        return None
    except Exception as e:
        st.error(f"'{file_path}' の読み込み中にエラーが発生しました: {e}")
        return None

@st.cache_data
def load_details_data(file_path, hinban_col):
    """詳細情報Excelファイルを読み込み、品番列をクリーニングしてDataFrameを返す"""
    try:
        df = pd.read_excel(file_path)
        if hinban_col in df.columns:
            df[hinban_col] = df[hinban_col].fillna('').astype(str).str.strip()
        else:
            st.error(f"エラー: 詳細ファイル '{file_path}' に '{hinban_col}' 列が見つかりません。")
            return None
        return df
    except FileNotFoundError:
        st.error(f"エラー: ファイル '{file_path}' が見つかりません。")
        st.info(f"スクリプトと同じディレクトリに '{file_path}' ファイルを配置してください。")
        return None
    except Exception as e:
        st.error(f"'{file_path}' の読み込み中にエラーが発生しました: {e}")
        return None

@st.cache_data
def load_comment_data(file_path, hinban_col):
    """コメント情報Excelファイルを読み込み、品番列をクリーニングしてDataFrameを返す"""
    try:
        df = pd.read_excel(file_path)
        if hinban_col in df.columns:
            df[hinban_col] = df[hinban_col].fillna('').astype(str).str.strip()
        else:
            st.error(f"エラー: コメントファイル '{file_path}' に '{hinban_col}' 列が見つかりません。")
            return None
        return df
    except FileNotFoundError:
        st.error(f"エラー: ファイル '{file_path}' が見つかりません。")
        st.info(f"スクリプトと同じディレクトリに '{file_path}' ファイルを配置してください。")
        return None
    except Exception as e:
        st.error(f"'{file_path}' の読み込み中にエラーが発生しました: {e}")
        return None

# --- 画像表示補助関数 ---
def display_images(df_to_display, filename_col, image_folder, num_cols, displayed_filenames_set, missing_files_list):
    """DataFrame内の画像を表示し、表示したファイル名と見つからないファイル名を更新する"""
    cols = st.columns(num_cols)
    image_count = 0
    col_idx_counter = 0 # カラムのインデックスを管理するカウンター

    for index, row in df_to_display.iterrows():
        filename = row[filename_col]
        is_filename_valid = filename and isinstance(filename, str) and filename.strip()

        if not is_filename_valid:
            missing_msg = f"無効なファイル名 (行 {index+2})"
            if missing_msg not in missing_files_list:
                 missing_files_list.append(missing_msg)
            continue

        # ★ 既に他の場所で表示されているファイル名はスキップ
        if filename in displayed_filenames_set:
            continue

        image_path = os.path.join(image_folder, filename)
        filename_without_ext = pathlib.Path(filename).stem

        col_index = col_idx_counter % num_cols
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path)
                with cols[col_index]:
                    st.image(image, caption=filename_without_ext, use_container_width=True)
                image_count += 1
                col_idx_counter += 1 # 画像表示成功時にカラムカウンターを進める
                displayed_filenames_set.add(filename) # 表示済みセットに追加
            else:
                # 画像ファイルが見つからない場合
                if filename not in missing_files_list:
                    missing_files_list.append(filename)
                # 見つからない場合でも表示済みセットに追加（次回以降の重複処理を防ぐため）
                displayed_filenames_set.add(filename)
                # 見つからない場合、カラムを消費するかどうかはデザインによる
                # ここではカラムを消費しない（次の有効な画像が同じカラムに入る）

        except Exception as e:
             # 画像ファイルを開く際のエラーなど
             with cols[col_index]:
                 st.warning(f"{filename}\n画像表示エラー:\n{e}")
             col_idx_counter += 1 # エラーでもカラムカウンターを進める
             displayed_filenames_set.add(filename) # エラーでも表示済みセットに追加

    return image_count # この関数呼び出しで新たに追加表示された画像の数を返す


# --- メイン処理 ---
st.markdown("### チェア画像の検索")

# --- 各データファイルを読み込み ---
# ★ 修正: load_classification_data に flag_col を渡す
df_classification = load_classification_data(EXCEL_FILE, FILENAME_COLUMN, COLUMNS_TO_FILTER, COLOR_FLAG_COLUMN)
df_details = load_details_data(DETAILS_EXCEL_FILE, DETAILS_HINBAN_COLUMN)
df_comment = load_comment_data(COMMENT_EXCEL_FILE, COMMENT_HINBAN_COLUMN)

if df_classification is not None:
    st.sidebar.header("検索条件")

    filter_values = {}
    for col in COLUMNS_TO_FILTER:
        if col in df_classification.columns:
            unique_options = sorted([opt for opt in df_classification[col].unique().tolist() if opt])
            options = ["すべて"] + unique_options
            session_key = f'select_{col}'
            if session_key not in st.session_state:
                st.session_state[session_key] = "すべて"
            filter_values[col] = st.sidebar.selectbox(f"{col}:", options, key=session_key)
        else:
            st.sidebar.warning(f"警告: Excelファイルに '{col}' 列が見つかりません。")
            filter_values[col] = "すべて"

    search_button = st.sidebar.button("検索")

    st.markdown("#### 検索結果")

    # --- セッションステート初期化 ---
    if 'search_clicked' not in st.session_state:
        st.session_state.search_clicked = False
    if 'filtered_df_for_display' not in st.session_state:
        st.session_state.filtered_df_for_display = pd.DataFrame()
    if 'initial_hinban_list' not in st.session_state:
        st.session_state.initial_hinban_list = []
    if 'selected_hinban' not in st.session_state:
        st.session_state.selected_hinban = "品番を選択してください"
    if 'missing_files_list' not in st.session_state:
        st.session_state.missing_files_list = []
    if 'all_displayed_filenames' not in st.session_state:
        st.session_state.all_displayed_filenames = set()
    # ★ 追加: 「更に見る」関連のセッションステート
    if 'show_more_clicked' not in st.session_state:
        st.session_state.show_more_clicked = False
    if 'more_df' not in st.session_state: # 「更に見る」で表示対象となるDataFrame
        st.session_state.more_df = pd.DataFrame()
    if 'more_hinban_list' not in st.session_state: # 「更に見る」で追加された品番リスト
        st.session_state.more_hinban_list = []
    if 'total_initial_images_displayed' not in st.session_state: # 初期表示された画像数
        st.session_state.total_initial_images_displayed = 0
    if 'total_more_images_displayed' not in st.session_state: # 追加表示された画像数
        st.session_state.total_more_images_displayed = 0


    # --- 検索ボタン押下時の処理 ---
    if search_button:
        st.session_state.search_clicked = True
        filtered_df = df_classification.copy()
        for col, selected_value in filter_values.items():
            if selected_value != "すべて" and col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col] == selected_value]

        st.session_state.filtered_df_for_display = filtered_df
        hinban_list_with_ext = filtered_df[FILENAME_COLUMN].tolist()
        hinban_list_no_ext = sorted(list(set([pathlib.Path(f).stem for f in hinban_list_with_ext if f and isinstance(f, str) and f.strip()])))
        st.session_state.initial_hinban_list = hinban_list_no_ext

        # ★ リセット処理
        st.session_state.selected_hinban = "品番を選択してください"
        st.session_state.missing_files_list = []
        st.session_state.all_displayed_filenames = set()
        st.session_state.show_more_clicked = False
        st.session_state.more_df = pd.DataFrame()
        st.session_state.more_hinban_list = []
        st.session_state.total_initial_images_displayed = 0
        st.session_state.total_more_images_displayed = 0


    # --- 検索結果表示エリア ---
    if st.session_state.search_clicked:
        initial_df_display = st.session_state.filtered_df_for_display

        if not initial_df_display.empty:
            st.write(f"検索条件に {len(initial_df_display)} 件のデータが該当しました。")

            # --- ① 最初の画像表示 ---
            # 検索ボタンが押された直後のみ実行し、表示数を記録
            if not st.session_state.all_displayed_filenames: # まだ何も表示されていない場合
                st.write("**検索結果の画像:**")
                initial_newly_displayed_count = display_images(
                    initial_df_display,
                    FILENAME_COLUMN,
                    IMAGE_FOLDER,
                    NUM_DISPLAY_COLUMNS,
                    st.session_state.all_displayed_filenames, # 更新されるセットを渡す
                    st.session_state.missing_files_list      # 更新されるリストを渡す
                )
                st.session_state.total_initial_images_displayed = initial_newly_displayed_count
                if initial_newly_displayed_count > 0:
                    st.write(f"（{initial_newly_displayed_count} 件の画像を表示）")
            else:
                # 再描画時（例：「更に見る」クリック後など）は、記録された数を表示
                 if st.session_state.total_initial_images_displayed > 0:
                    st.write("**検索結果の画像:**")
                    # ここで再度 display_images を呼ぶと重複する可能性があるため、表示済みのメッセージのみ出す
                    st.write(f"（{st.session_state.total_initial_images_displayed} 件の画像を表示済み）")


            # --- ② 「更に見る」ボタンと処理 ---
            st.markdown("---")
            # 「更に見る」ボタンは、初期検索結果がある場合に表示
            if not initial_df_display.empty:
                show_more_button = st.button("更に表示する（赤/黄/白フラグのある項目）")

                if show_more_button:
                    st.session_state.show_more_clicked = True # ボタンが押されたフラグ
                    st.session_state.total_more_images_displayed = 0 # 追加表示数をリセット

                    if COLOR_FLAG_COLUMN in df_classification.columns:
                        # 元の全データからフラグが1のものを抽出
                        more_df_candidates = df_classification[df_classification[COLOR_FLAG_COLUMN] == 1].copy()

                        # ★ 既に表示されているファイル名は除外する
                        more_df_to_consider = more_df_candidates[~more_df_candidates[FILENAME_COLUMN].isin(st.session_state.all_displayed_filenames)]

                        if not more_df_to_consider.empty:
                            st.session_state.more_df = more_df_to_consider # 表示候補を保存
                            more_hinban_list_with_ext = st.session_state.more_df[FILENAME_COLUMN].tolist()
                            more_hinban_list_no_ext = sorted(list(set([pathlib.Path(f).stem for f in more_hinban_list_with_ext if f and isinstance(f, str) and f.strip()])))
                            # 既存のリストと重複しないように追加
                            new_hinbans = [h for h in more_hinban_list_no_ext if h not in st.session_state.initial_hinban_list and h not in st.session_state.more_hinban_list]
                            st.session_state.more_hinban_list.extend(new_hinbans)
                            st.session_state.more_hinban_list.sort() # 追加後にソート

                            # ここではDataFrameを保存するだけで、表示は次のステップで行う
                            # st.experimental_rerun() # 即時再描画が必要な場合

                        else:
                            st.info("「更に表示する」条件に合う、まだ表示されていない項目はありません。")
                            st.session_state.more_df = pd.DataFrame() # 候補がないので空にする
                            # more_hinban_list はリセットしない（過去に追加されたものは保持）
                    else:
                        st.warning(f"'{COLOR_FLAG_COLUMN}' 列がExcelファイルに存在しないため、「更に見る」機能は利用できません。")
                        st.session_state.more_df = pd.DataFrame()


            # --- ③ 「更に見る」がクリックされた後の追加画像表示 ---
            # show_more_clicked フラグが True で、表示候補の more_df が空でない場合に実行
            if st.session_state.show_more_clicked and not st.session_state.more_df.empty:
                st.write(f"**「更に表示する」結果 ({len(st.session_state.more_df)} 件の候補):**")
                # display_images を呼び出して追加表示
                more_newly_displayed_count = display_images(
                    st.session_state.more_df,
                    FILENAME_COLUMN,
                    IMAGE_FOLDER,
                    NUM_DISPLAY_COLUMNS,
                    st.session_state.all_displayed_filenames, # 更新されるセットを渡す
                    st.session_state.missing_files_list      # 更新されるリストを渡す
                )
                # ボタンクリック後の最初の描画で表示数を記録
                if st.session_state.total_more_images_displayed == 0:
                     st.session_state.total_more_images_displayed = more_newly_displayed_count

                if st.session_state.total_more_images_displayed > 0:
                    st.write(f"（うち {st.session_state.total_more_images_displayed} 件の画像を追加表示）")
                # else:
                #     st.info("追加で表示できる画像はありませんでした。（ファイルが存在しないか、既に表示済み）")


            # --- ④ 見つからないファイルリストの表示 ---
            if st.session_state.missing_files_list:
                st.warning(f"注意: {len(st.session_state.missing_files_list)} 件のデータに対応する画像ファイルが見つからないか、ファイル名が無効です。")
                with st.expander("詳細（見つからない/無効なファイル名）"):
                    for missing_item in st.session_state.missing_files_list:
                        st.write(f"- {missing_item}")


            # --- ⑤ 品番選択セレクトボックスの作成 ---
            st.markdown("---")
            st.markdown("#### 詳細情報の表示")

            # ★ 修正: initial_hinban_list と more_hinban_list を結合して選択肢を作成
            combined_hinban_list = sorted(list(set(st.session_state.initial_hinban_list + st.session_state.more_hinban_list)))

            if combined_hinban_list:
                hinban_options = ["品番を選択してください"] + combined_hinban_list

                current_selection_index = 0
                if st.session_state.selected_hinban in hinban_options:
                    current_selection_index = hinban_options.index(st.session_state.selected_hinban)

                selected_hinban = st.selectbox(
                    "品番を選択:",
                    options=hinban_options,
                    index=current_selection_index,
                    key="hinban_selector"
                )

                if selected_hinban != st.session_state.selected_hinban:
                    st.session_state.selected_hinban = selected_hinban
                    # st.experimental_rerun() # 選択変更時に即時再描画する場合

                # --- ⑥ 品番選択後の詳細表示 ---
                if st.session_state.selected_hinban != "品番を選択してください":
                    selected_hinban_value = st.session_state.selected_hinban

                    # --- chair_all.xlsx の情報表示 ---
                    if df_details is not None:
                        details_rows = df_details[df_details[DETAILS_HINBAN_COLUMN] == selected_hinban_value]
                        if not details_rows.empty:
                            st.write(f"**品番: {selected_hinban_value} の詳細情報 ({DETAILS_EXCEL_FILE})**")
                            st.dataframe(details_rows.reset_index(drop=True))
                        else:
                            st.info(f"ファイル '{DETAILS_EXCEL_FILE}' 内に品番 '{selected_hinban_value}' の詳細情報は見つかりませんでした。")
                    else:
                        st.error(f"詳細情報ファイル '{DETAILS_EXCEL_FILE}' を読み込めませんでした。")

                    # --- comment.xlsx の情報表示 ---
                    st.markdown("---")
                    if df_comment is not None:
                        comment_rows = df_comment[df_comment[COMMENT_HINBAN_COLUMN] == selected_hinban_value]
                        if not comment_rows.empty:
                            st.write(f"**品番: {selected_hinban_value} のコメント情報 ({COMMENT_EXCEL_FILE})**")
                            display_comment = comment_rows.iloc[[0]].reset_index(drop=True)
                            st.table(display_comment)
                        else:
                            st.info(f"ファイル '{COMMENT_EXCEL_FILE}' 内に品番 '{selected_hinban_value}' のコメント情報は見つかりませんでした。")
                    else:
                        st.warning(f"コメント情報ファイル '{COMMENT_EXCEL_FILE}' が読み込めなかったため、コメント情報は表示できません。")

                else: # 品番が選択されていない場合
                    st.info("上のセレクトボックスから品番を選択すると、詳細情報とコメント情報が表示されます。")
            else:
                 if not st.session_state.missing_files_list:
                     st.info("表示できる品番がありません。")

        else: # 最初の検索結果が0件の場合
            st.info("指定された条件に一致するデータはありませんでした。")
            # 検索状態をリセット
            st.session_state.search_clicked = False
            st.session_state.filtered_df_for_display = pd.DataFrame()
            st.session_state.initial_hinban_list = []
            st.session_state.selected_hinban = "品番を選択してください"
            st.session_state.missing_files_list = []
            st.session_state.all_displayed_filenames = set()
            st.session_state.show_more_clicked = False
            st.session_state.more_df = pd.DataFrame()
            st.session_state.more_hinban_list = []
            st.session_state.total_initial_images_displayed = 0
            st.session_state.total_more_images_displayed = 0


    else: # 検索ボタンがまだ押されていない場合
        st.info("左側のサイドバーで条件を選択し、「検索」ボタンを押してください。")

else: # メインの分類ファイルが読み込めなかった場合
    st.warning(f"メインのデータファイル '{EXCEL_FILE}' を読み込めませんでした。処理を続行できません。")


# --- 修理価格一覧 & 注意点 (変更なし) ---
st.markdown("---")
st.subheader("修理価格一覧")
if os.path.exists(PRICE_LIST_EXCEL_PATH):
    try:
        df_prices = pd.read_excel(PRICE_LIST_EXCEL_PATH)
        df_prices_display = df_prices.fillna("").astype(str)
        st.table(df_prices_display)
    except FileNotFoundError:
        st.error(f"❌ 修理価格一覧Excelファイルが見つかりません: {PRICE_LIST_EXCEL_PATH}")
    except Exception as e:
        st.error(f"❌ 修理価格一覧Excelファイルの読み込みエラー: {e}")
else:
    st.warning(f"⚠️ 修理価格一覧Excelファイルが見つかりません: {PRICE_LIST_EXCEL_PATH}")

st.markdown("---")
st.subheader("注意点")
if os.path.exists(NOTES_EXCEL_PATH):
    try:
        df_notes = pd.read_excel(NOTES_EXCEL_PATH)
        df_notes_display = df_notes.fillna("").astype(str)
        st.table(df_notes_display)
    except Exception as e:
        st.error(f"❌ 注意点Excelファイル ({NOTES_EXCEL_PATH}) の読み込みエラー: {e}")
else:
    st.warning(f"⚠️ 注意点Excelファイルが見つかりません: {NOTES_EXCEL_PATH}")


# --- (オプション) デバッグ用 ---
# (コメント、分類、詳細の元のデータ表示は変更なし)
if st.sidebar.checkbox("元のデータを表示 (コメント)"):
     if df_comment is not None:
         st.subheader(f"元のデータ ({COMMENT_EXCEL_FILE})")
         st.dataframe(df_comment)

if st.sidebar.checkbox("元のデータを表示 (分類)"):
    if df_classification is not None:
        st.subheader(f"元のデータ ({EXCEL_FILE})")
        st.dataframe(df_classification)
if st.sidebar.checkbox("元のデータを表示 (詳細)"):
     if df_details is not None:
         st.subheader(f"元のデータ ({DETAILS_EXCEL_FILE})")
         st.dataframe(df_details)

# デバッグ用に「更に見る」で抽出されたデータも表示
if st.session_state.get('show_more_clicked', False):
    if st.sidebar.checkbox("「更に見る」候補データを表示"):
        st.subheader("「更に見る」候補データ (表示済み除く)")
        st.dataframe(st.session_state.more_df)

if st.session_state.get('search_clicked', False):
     if st.sidebar.checkbox("絞り込み後のデータを表示 (分類)"):
        st.subheader("絞り込み後のデータ (分類)")
        st.dataframe(st.session_state.filtered_df_for_display)

# import streamlit as st
# import pandas as pd
# import os
# from PIL import Image
# import pathlib

# # --- ページ設定を最初に行う ---
# st.set_page_config(layout="wide")

# # --- 画像とキャプションのスタイル調整 ---
# # (スタイル部分は変更なし)
# style = """
# <style>
# /* 画像要素自体のスタイル */
# div[data-testid="stImage"] img {
#     max-height: 200px;  /* ★ 画像の最大高さを指定 (この値を調整) */
#     object-fit: contain; /* アスペクト比を保ちコンテナ内に収める */
#     display: block;      /* 中央揃えのためにブロック要素に */
#     margin-left: auto;   /* 画像を中央揃え */
#     margin-right: auto;  /* 画像を中央揃え */
#     width: auto;         /* 高さに合わせて幅を自動調整 */
# }
# /* キャプションのスタイル */
# figcaption {
#     font-size: 18px !important; /* キャプションの文字サイズ */
#     color: #555;             /* キャプションの色 */
#     text-align: center;      /* キャプションを中央揃え */
# }
# </style>
# """
# st.markdown(style, unsafe_allow_html=True)
# # --- スタイル調整ここまで ---


# # --- 定数 ---
# EXCEL_FILE = "chair_classification_results.xlsx"
# DETAILS_EXCEL_FILE = "chair_all.xlsx"
# PRICE_LIST_EXCEL_PATH = "修理価格一覧app用.xlsx"
# COMMENT_EXCEL_FILE = "comment.xlsx" # ★ 追加: コメント情報ファイル
# NOTES_EXCEL_PATH = "修理注意点.xlsx" 
# IMAGE_FOLDER = "images"
# COLUMNS_TO_FILTER = ["背のデザイン", "座面", "肘"]
# FILENAME_COLUMN = "ファイル名" # chair_classification_results.xlsx のファイル名列
# DETAILS_HINBAN_COLUMN = "品番" # chair_all.xlsx の品番列名
# COMMENT_HINBAN_COLUMN = "品番" # ★ 追加: comment.xlsx の品番列名 (実際の列名に合わせてください)
# NUM_DISPLAY_COLUMNS = 6

# # --- データ読み込み関数 ---
# @st.cache_data
# def load_classification_data(file_path, filename_col, cols_to_clean):
#     """分類情報Excelファイルを読み込み、指定列をクリーニングしてDataFrameを返す"""
#     try:
#         df = pd.read_excel(file_path)
#         if filename_col in df.columns:
#             df[filename_col] = df[filename_col].fillna('').astype(str).str.strip()
#         else:
#             st.warning(f"警告: ファイル '{file_path}' に '{filename_col}' 列が見つかりません。")
#             return None
#         for col in cols_to_clean:
#             if col in df.columns:
#                 df[col] = df[col].fillna('').astype(str).str.strip().str.replace('　', ' ', regex=False).str.strip()
#         return df
#     except FileNotFoundError:
#         st.error(f"エラー: ファイル '{file_path}' が見つかりません。")
#         st.info(f"スクリプトと同じディレクトリに '{file_path}' ファイルを配置してください。")
#         return None
#     except Exception as e:
#         st.error(f"'{file_path}' の読み込み中にエラーが発生しました: {e}")
#         return None

# @st.cache_data
# def load_details_data(file_path, hinban_col):
#     """詳細情報Excelファイルを読み込み、品番列をクリーニングしてDataFrameを返す"""
#     try:
#         df = pd.read_excel(file_path)
#         if hinban_col in df.columns:
#             df[hinban_col] = df[hinban_col].fillna('').astype(str).str.strip()
#             # ★ 必要に応じてクリーニング処理を追加 ★
#         else:
#             st.error(f"エラー: 詳細ファイル '{file_path}' に '{hinban_col}' 列が見つかりません。")
#             return None
#         return df
#     except FileNotFoundError:
#         st.error(f"エラー: ファイル '{file_path}' が見つかりません。")
#         st.info(f"スクリプトと同じディレクトリに '{file_path}' ファイルを配置してください。")
#         return None
#     except Exception as e:
#         st.error(f"'{file_path}' の読み込み中にエラーが発生しました: {e}")
#         return None

# # ★ 追加: コメントデータ読み込み関数 ---
# @st.cache_data
# def load_comment_data(file_path, hinban_col):
#     """コメント情報Excelファイルを読み込み、品番列をクリーニングしてDataFrameを返す"""
#     try:
#         df = pd.read_excel(file_path)
#         if hinban_col in df.columns:
#             df[hinban_col] = df[hinban_col].fillna('').astype(str).str.strip()
#             # ★ 必要に応じてクリーニング処理を追加 ★
#             # 例: df[hinban_col] = df[hinban_col].str.replace('#', '', regex=False)
#         else:
#             st.error(f"エラー: コメントファイル '{file_path}' に '{hinban_col}' 列が見つかりません。")
#             return None
#         return df
#     except FileNotFoundError:
#         # コメントファイルはオプション扱いとし、エラーではなく警告にする場合
#         # st.warning(f"警告: コメントファイル '{file_path}' が見つかりません。コメント情報は表示されません。")
#         st.error(f"エラー: ファイル '{file_path}' が見つかりません。")
#         st.info(f"スクリプトと同じディレクトリに '{file_path}' ファイルを配置してください。")
#         return None
#     except Exception as e:
#         st.error(f"'{file_path}' の読み込み中にエラーが発生しました: {e}")
#         return None

# # --- メイン処理 ---
# st.markdown("### チェア画像の検索")

# # --- 各データファイルを読み込み ---
# df_classification = load_classification_data(EXCEL_FILE, FILENAME_COLUMN, COLUMNS_TO_FILTER)
# df_details = load_details_data(DETAILS_EXCEL_FILE, DETAILS_HINBAN_COLUMN)
# df_comment = load_comment_data(COMMENT_EXCEL_FILE, COMMENT_HINBAN_COLUMN) # ★ 追加

# if df_classification is not None:
#     st.sidebar.header("検索条件")

#     filter_values = {}
#     for col in COLUMNS_TO_FILTER:
#         if col in df_classification.columns:
#             unique_options = sorted([opt for opt in df_classification[col].unique().tolist() if opt])
#             options = ["すべて"] + unique_options
#             if f'select_{col}' not in st.session_state:
#                 st.session_state[f'select_{col}'] = "すべて"
#             filter_values[col] = st.sidebar.selectbox(f"{col}:", options, key=f'select_{col}')
#         else:
#             st.sidebar.warning(f"警告: Excelファイルに '{col}' 列が見つかりません。")
#             filter_values[col] = "すべて"

#     search_button = st.sidebar.button("検索")

#     st.markdown("#### 検索結果")

#     # --- セッションステート初期化 (変更なし) ---
#     if 'search_clicked' not in st.session_state:
#         st.session_state.search_clicked = False
#     if 'filtered_hinban_list' not in st.session_state:
#         st.session_state.filtered_hinban_list = []
#     if 'filtered_df_for_display' not in st.session_state:
#         st.session_state.filtered_df_for_display = pd.DataFrame()
#     if 'selected_hinban' not in st.session_state:
#         st.session_state.selected_hinban = "品番を選択してください"
#     if 'missing_files_list' not in st.session_state:
#         st.session_state.missing_files_list = []

#     # --- 検索ボタン押下時の処理 (変更なし) ---
#     if search_button:
#         st.session_state.search_clicked = True
#         filtered_df = df_classification.copy()
#         for col, selected_value in filter_values.items():
#             if selected_value != "すべて" and col in filtered_df.columns:
#                 filtered_df = filtered_df[filtered_df[col] == selected_value]

#         st.session_state.filtered_df_for_display = filtered_df
#         hinban_list_with_ext = filtered_df[FILENAME_COLUMN].tolist()
#         hinban_list_no_ext = [pathlib.Path(f).stem for f in hinban_list_with_ext if f and isinstance(f, str)]
#         st.session_state.filtered_hinban_list = hinban_list_no_ext
#         st.session_state.selected_hinban = "品番を選択してください"
#         st.session_state.missing_files_list = []

#     # --- 検索結果表示エリア ---
#     if st.session_state.search_clicked:
#         filtered_df_display = st.session_state.filtered_df_for_display

#         if not filtered_df_display.empty:
#             st.write(f"{len(filtered_df_display)} 件のデータが見つかりました。")

#             # --- 画像表示 (変更なし) ---
#             cols = st.columns(NUM_DISPLAY_COLUMNS)
#             image_count = 0
#             displayed_filenames = set()
#             missing_files_current_search = []

#             for index, row in filtered_df_display.iterrows():
#                 filename = row[FILENAME_COLUMN]
#                 is_filename_valid = filename and isinstance(filename, str)
#                 if not is_filename_valid:
#                     missing_files_current_search.append(f"無効なファイル名 (行 {index+2})")
#                     continue
#                 if filename in displayed_filenames:
#                     continue

#                 image_path = os.path.join(IMAGE_FOLDER, filename)
#                 filename_without_ext = pathlib.Path(filename).stem

#                 if os.path.exists(image_path):
#                     try:
#                         image = Image.open(image_path)
#                         col_index = image_count % NUM_DISPLAY_COLUMNS
#                         with cols[col_index]:
#                             st.image(image, caption=filename_without_ext, use_container_width=True)
#                         image_count += 1
#                         displayed_filenames.add(filename)
#                     except Exception as e:
#                          col_index = image_count % NUM_DISPLAY_COLUMNS
#                          with cols[col_index]:
#                              st.warning(f"{filename}\n画像表示エラー:\n{e}")
#                          image_count += 1
#                          displayed_filenames.add(filename)
#                 else:
#                     missing_files_current_search.append(filename)
#                     displayed_filenames.add(filename)

#             st.session_state.missing_files_list = missing_files_current_search

#             if image_count > 0:
#                  st.write(f"（うち {image_count} 件の画像を表示しました）")

#             if st.session_state.missing_files_list:
#                 st.warning(f"注意: {len(st.session_state.missing_files_list)} 件のデータに対応する画像ファイルが見つからないか、ファイル名が無効です。")
#                 with st.expander("詳細（見つからない/無効なファイル名）"):
#                     st.write(st.session_state.missing_files_list)

#             # --- ① 品番選択セレクトボックスの作成 (変更なし) ---
#             st.markdown("---")
#             st.markdown("#### 詳細情報の表示")
#             unique_hinban_list = sorted(list(set(st.session_state.filtered_hinban_list)))
#             if unique_hinban_list:
#                 hinban_options = ["品番を選択してください"] + unique_hinban_list
#                 selected_hinban = st.selectbox(
#                     "品番を選択:",
#                     options=hinban_options,
#                     index=hinban_options.index(st.session_state.selected_hinban),
#                     key="hinban_selector"
#                 )
#                 if selected_hinban != st.session_state.selected_hinban:
#                     st.session_state.selected_hinban = selected_hinban
#                     # st.experimental_rerun()

#                 # --- ② 品番選択後の詳細表示 ---
#                 if st.session_state.selected_hinban != "品番を選択してください":
#                     selected_hinban_value = st.session_state.selected_hinban

#                     # --- chair_all.xlsx の情報表示 ---
#                     if df_details is not None:
#                         details_rows = df_details[df_details[DETAILS_HINBAN_COLUMN] == selected_hinban_value]
#                         if not details_rows.empty:
#                             st.write(f"**品番: {selected_hinban_value} の詳細情報 ({DETAILS_EXCEL_FILE})**")
#                             st.dataframe(details_rows)
#                         else:
#                             st.info(f"ファイル '{DETAILS_EXCEL_FILE}' 内に品番 '{selected_hinban_value}' の詳細情報は見つかりませんでした。")
#                     else:
#                         st.error(f"詳細情報ファイル '{DETAILS_EXCEL_FILE}' を読み込めませんでした。")

#                     # --- ★ 追加: comment.xlsx の情報表示 ---
#                     st.markdown("---") # 区切り線を追加
#                     if df_comment is not None:
#                         # comment.xlsx の品番列と選択された品番を比較
#                         comment_rows = df_comment[df_comment[COMMENT_HINBAN_COLUMN] == selected_hinban_value]
#                         if not comment_rows.empty:
#                             st.write(f"**品番: {selected_hinban_value} のコメント情報 ({COMMENT_EXCEL_FILE})**")
#                             # コメント情報は通常1行と想定される場合が多いが、複数行対応のため dataframe を使用
#                             st.table(comment_rows.iloc[0].transpose())
#                         else:
#                             # コメントがないのは通常なので info レベルで表示
#                             st.info(f"ファイル '{COMMENT_EXCEL_FILE}' 内に品番 '{selected_hinban_value}' のコメント情報は見つかりませんでした。")
#                     else:
#                         # コメントファイル自体が読み込めていない場合のエラー表示
#                         st.warning(f"コメント情報ファイル '{COMMENT_EXCEL_FILE}' が読み込めなかったため、コメント情報は表示できません。")
#                         # 必要であれば load_comment_data のエラーメッセージをここで再表示してもよい

#                 else: # 品番が選択されていない場合
#                     st.info("上のセレクトボックスから品番を選択すると、詳細情報とコメント情報が表示されます。")
#             else:
#                  if not st.session_state.missing_files_list:
#                      st.info("表示できる品番がありません。")

#         else: # 検索結果が0件の場合
#             st.info("指定された条件に一致するデータはありませんでした。")
#             st.session_state.search_clicked = False
#             st.session_state.filtered_hinban_list = []
#             st.session_state.filtered_df_for_display = pd.DataFrame()
#             st.session_state.selected_hinban = "品番を選択してください"
#             st.session_state.missing_files_list = []

#     else: # 検索ボタンがまだ押されていない場合
#         st.info("左側のサイドバーで条件を選択し、「検索」ボタンを押してください。")

# else: # メインの分類ファイルが読み込めなかった場合
#     st.warning(f"メインのデータファイル '{EXCEL_FILE}' を読み込めませんでした。処理を続行できません。")

# # --- 修理価格一覧 ---
# st.markdown("---")
# st.subheader("修理価格一覧")
# if os.path.exists(PRICE_LIST_EXCEL_PATH):
#     try:
#         df_prices = pd.read_excel(PRICE_LIST_EXCEL_PATH)
#         # NaNを空文字に置換し、全列を文字列に変換して表示
#         df_prices_display = df_prices.fillna("").astype(str)
#         st.table(df_prices_display)
#     except FileNotFoundError:
#         st.error(f"❌ 修理価格一覧Excelファイルが見つかりません: {PRICE_LIST_EXCEL_PATH}")
#     except Exception as e:
#         st.error(f"❌ 修理価格一覧Excelファイルの読み込みエラー: {e}")
# else:
#     st.warning(f"⚠️ 修理価格一覧Excelファイルが見つかりません: {PRICE_LIST_EXCEL_PATH}")

# # --- 注意点 ---
# st.markdown("---")
# st.subheader("注意点")
# if os.path.exists(NOTES_EXCEL_PATH):
#     try:
#         df_notes = pd.read_excel(NOTES_EXCEL_PATH)
#         # NaNを空文字に置換し、全列を文字列に変換して表示
#         df_notes_display = df_notes.fillna("").astype(str)
#         st.table(df_notes_display) # テーブル形式で表示
#     except Exception as e:
#         st.error(f"❌ 注意点Excelファイル ({NOTES_EXCEL_PATH}) の読み込みエラー: {e}")
# else:
#     st.warning(f"⚠️ 注意点Excelファイルが見つかりません: {NOTES_EXCEL_PATH}")

# # --- (オプション) デバッグ用 ---
# if st.sidebar.checkbox("元のデータを表示 (コメント)"):
#      if df_comment is not None:
#          st.subheader(f"元のデータ ({COMMENT_EXCEL_FILE})")
#          st.dataframe(df_comment)

# if st.sidebar.checkbox("元のデータを表示 (分類)"):
#     if df_classification is not None:
#         st.subheader(f"元のデータ ({EXCEL_FILE})")
#         st.dataframe(df_classification)
# if st.sidebar.checkbox("元のデータを表示 (詳細)"):
#      if df_details is not None:
#          st.subheader(f"元のデータ ({DETAILS_EXCEL_FILE})")
#          st.dataframe(df_details)

# if st.session_state.get('search_clicked', False):
#      if st.sidebar.checkbox("絞り込み後のデータを表示 (分類)"):
#         st.subheader("絞り込み後のデータ (分類)")
#         st.dataframe(st.session_state.filtered_df_for_display)

