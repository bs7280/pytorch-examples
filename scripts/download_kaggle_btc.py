import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    "novandraanugrah/bitcoin-historical-datasets-2018-2024",
    path="../data/bitcoin-historical-datasets/",
    unzip=True,
)
