from dataset_utils import get_dataset, get_textual_df, encode_item_tag, EInvoiceDataCollator


train_df = get_textual_df(data_args.train_file)
train_df, num_labels = encode_item_tag(train_df)

train_dataset = get_dataset(train_df)
