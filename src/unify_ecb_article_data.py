import polars as pl
from loguru import logger
from pathlib import Path

FILE_DIR = Path(__file__).parent
DATA_DIR = FILE_DIR.parent / 'data' / 'blogs_articles'

def list_cleaning_year(df:pl.DataFrame, list_col: str = 'filter_value'): 
    df_copy = df.clone()
    df_copy = df_copy.with_row_index(name="id", offset=0)
    temp = df_copy[['id', list_col]].to_dicts()
    temp_new = []
    for d in temp:
        cleaned_val = [int(s.strip("[]")) for s in d[list_col]]
        temp_new.append({
            "id": d["id"],
            list_col: cleaned_val
        })

    df_temp_new = pl.DataFrame(temp_new)

    df_copy = df_copy.drop(list_col)
    df_copy = df_copy.join(df_temp_new, on = 'id')
    df_copy = df_copy.drop('id')
    return df_copy


def type_of_document(df:pl.DataFrame): 
    df = df.with_columns([
        pl.col("url").str.extract(r"\.([^.]+)$").str.to_lowercase().alias("suffix")
        ])
    
    counts = {d["suffix"]: d["count"] for d in df['suffix'].value_counts().sort('count', descending = True).to_dicts()}
    logger.info(f"Types of Documents: {counts}")
    return df

def create_filename_col(df:pl.DataFrame): 
    df = df.with_columns([
        pl.col("url").str.extract(r"([^/]+)$").alias("filename"),
    ])

    dupes_per_suffix = (
        df.group_by("suffix")
        .agg((pl.count("filename") - pl.col("filename").n_unique()).alias("duplicate_count"))
    )

    no_dups = True
    for row in dupes_per_suffix.iter_rows(named=True):
        if row['duplicate_count'] != 0: 
            no_dups = False
            logger.info(f"Suffix {row['suffix']}: {row['duplicate_count']} duplicates in column filename")
    if no_dups: 
        logger.info("No duplictaes in column filename.")

    return df

def base_categories(df:pl.DataFrame): 
    df = df.with_columns(
        pl.col("category")
        .cast(pl.Utf8)
        .str.replace(r"\s*-\s*.*$", "")   
        .str.strip_chars()
        .alias("category_base")
    )
    return df

def alignment_counts(topic_df, year_df):
    intersection = topic_df.join(topic_df, on="filename", how="inner").height
    only_in_topic = topic_df.join(year_df, on="filename", how="anti").height
    only_in_year = year_df.join(topic_df, on="filename", how="anti").height

    logger.info(f"filename in both: {intersection}")
    logger.info(f"Only in topic_df: {only_in_topic}")
    logger.info(f"Only in year_df: {only_in_year}")

def create_combined_dataset(topic_df:pl.DataFrame, year_df:pl.DataFrame): 
    year_df = year_df.drop('Unnamed: 0')
    logger.info("Dropped unnecessary column 'Unnamed: 0' from year_df.")
    year_df = list_cleaning_year(year_df)
    logger.info("Cleaned filter_value column in year_df.")

    logger.info(f"--- topic_df {topic_df.shape} ---")
    logger.info(f'columns: {topic_df.columns}')
    topic_df = type_of_document(topic_df)
    n_del = topic_df.shape[0]
    topic_df = topic_df.filter(pl.col('suffix') == "html")
    logger.info(f"Dropped {n_del - topic_df.shape[0]} rows with suffix not 'html'")
    topic_df = create_filename_col(topic_df)
    logger.info(f'Shape after all: {topic_df.shape}')

    logger.info(f"--- year_df {year_df.shape} ---")
    logger.info(f'columns: {year_df.columns}')
    year_df = type_of_document(year_df)
    n_del = year_df.shape[0]
    year_df = year_df.filter(pl.col('suffix') == "html")
    logger.info(f"Dropped {n_del - year_df.shape[0]} rows with suffix not 'html'")
    year_df = create_filename_col(year_df)
    logger.info(f'Shape after all: {year_df.shape}')

    alignment_counts(topic_df, year_df)

    result = pl.concat([topic_df, year_df], how = 'diagonal_relaxed')
    logger.info(f'Concatenated topic_df and year_df, shape: {result.shape}')
    result = base_categories(result)
    logger.info('Create category_base column from category column.')
    logger.info(f'Columns: {result.columns}')
    return result

def topic_mapping(df:pl.DataFrame):
    topics = df['filter_value'].to_list()
    topics_list = sorted(list(set([x for xs in topics for x in xs])))
    topic_mapping = dict(zip(range(len(topics_list)), topics_list))
    topic_mapping_ = dict(zip(topics_list, range(len(topics_list))))

    topic_labels = []
    topic_rows = df[['filename', 'filter_value']].rows(named=True)

    for row in topic_rows: 
        topic_labels.append({
            'filename': row['filename'], 
            'topic_label': [topic_mapping_.get(x) for x in row['filter_value']]
        })

    df = df.join(pl.DataFrame(topic_labels), on='filename', how = 'left')
    return pl.DataFrame({"topic_label": list(topic_mapping.keys()), "topic": list(topic_mapping.values())}), df

def split_df(df: pl.DataFrame, splits=(0.7, 0.15, 0.15), seed=42):
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n = len(df)
    train_end = int(splits[0] * n)
    val_end = train_end + int(splits[1] * n)
    train = df[:train_end]
    val = df[train_end:val_end]
    test = df[val_end:]
    logger.info(f'train: {len(train)}({round(len(train)/len(df), 4)}%)')
    logger.info(f'val: {len(val)}({round(len(val)/len(df), 4)}%)')
    logger.info(f'test: {len(test)}({round(len(test)/len(df), 4)}%)')    

    return train, val, test



if __name__ == '__main__': 

    save_path = DATA_DIR / 'combined_parsed_ecb_articles.parquet'
    
    year_path = DATA_DIR / "parsed_ecb_articles_by_year.parquet"
    topic_path = DATA_DIR / "parsed_ecb_articles_by_topic.parquet"
    
    year_df = pl.read_parquet(year_path)
    topic_df = pl.read_parquet(topic_path)

    combined_df = create_combined_dataset(topic_df, year_df)
    combined_df.write_parquet(save_path)
    logger.info(f'Saved resulting dataframe to {save_path.as_posix()}')

    # Splitting into Development Dataset (including train, validation & test) 
    # and the Rest (without labels)
    dev_df = combined_df.filter(pl.col('filter_type') == 'topic')
    topic_mappings, dev_df = topic_mapping(dev_df)
    topic_mappings.write_csv(DATA_DIR / 'topic_mappings.csv')
    logger.info('Saved topic mappings.')

    train, val, test = split_df(dev_df)
    train.write_parquet(DATA_DIR / 'train_topics.parquet')
    val.write_parquet(DATA_DIR / 'val_topics.parquet')
    test.write_parquet(DATA_DIR / 'test_topics.parquet')
    logger.info('Saved train, val & test.')
