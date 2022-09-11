import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional
# hf's setting
from transformers import (
        HfArgumentParser,
        TrainingArguments,
        DefaultDataCollator,
        Trainer,
)
# chinese nlp models
from transformers import BertTokenizerFast, AutoConfig
from models import BertForProductClassificationWithAgg
from dataset_utils import get_dataset, get_textual_df, encode_item_tag, EInvoiceDataCollator

@dataclass
class OurModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='ckiplab/bert-base-chinese')
    model_type: Optional[str] = field(default='bert-base-chinese')
    config_name: Optional[str] = field(default='ckiplab/bert-base-chinese')
    tokenizer_name: Optional[str] = field(default='bert-base-chinese')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

@dataclass
class OurDataArguments:
    # Huggingface's original arguments. 
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_length: int = field(default=32)
    train_file: Optional[str] = field(default='../data/2022.aigo.full.data.sample.csv')
    label_mapping_file: Optional[str] = field(default='category.mapping.tsv')

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Huggingface's original arguments. 
    output_dir: str = field(default='./models')
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=10000)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=2500)
    evaluation_strategy: Optional[str] = field(default='no')
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    logging_dir: Optional[str] = field(default='./logs')
    resume_from_checkpoint: Optional[str] = field(default=None)
    remove_unused_columns: Optional[bool] = field(default=False)

def main():

    # Parseing argument for huggingface packages
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ## Loading form hf dataset
    """
    Detail preprocessing in `dataset_utils`
    """
    train_df = get_textual_df(data_args.train_file)
    train_df, num_labels = encode_item_tag(train_df)

    train_dataset = get_dataset(train_df)

    # init
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    config_kwargs = {'num_labels': num_labels, 'output_hidden_states': True, }
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    model_kwargs = {'num_aspects': 10, 
                    'category_mapping': data_args.label_mapping_file,
                    'tokenizer': tokenizer}
    model = BertForProductClassificationWithAgg.from_pretrained(
            model_args.model_name_or_path, 
            config=config,
            **model_kwargs
    )

    # data collator
    data_collator = EInvoiceDataCollator(
            tokenizer=tokenizer, 
            padding=True,
            max_length = data_args.max_length,
            return_tensors='pt'
    )

    # Trainer
    trainer = Trainer(
            model=model, 
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator
    )
    
    # ***** strat training *****
    results = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    return results

if __name__ == '__main__':
    main()
