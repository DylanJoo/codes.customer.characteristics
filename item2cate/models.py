import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class BertForProductClassification(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config,)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # custormizer item2cate
        self.num_aspects = kwargs.pop('num_aspects', 10)
        self.lbl2cate = self._load_lbl_mapping(kwargs.pop('category_mapping', None))
        # Initialize weights and apply final processing
        self.post_init()

    def _load_lbl_mapping(self, path):
        mapping = {}
        with open(path, 'r') as f:
            for line in f:
                mapping[line.split('\t')[0]] = line.strip().split('\t')[1]
        return mapping

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # old sequence classification task
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # for evaluation and infernece/predict only
    def classify(
        self, 
        dataframe,
        batch_size=64,
        topk=10,
        output_files=None
    ):

        # 0) Prerequisite
        from model_utils import npmapping
        predictions = []
        softmax = nn.Softmax(dim=-1)

        # 1) Preprare dataset (batch input would be a better choice)
        from dataset_utils import get_dataset, EInvoiceDataCollator
        eval_dataset = get_dataset(dataframe)
        data_collator = EInvoiceDataCollator(
                tokenizer=tokenizer,
                padding=True,
                max_length=64,
                return_tensors='pt'
        )
        from torch.utils.data import DataLoader
        eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator
        )

        # batch prediction (classify)
        for b, batch in enumerate(eval_dataloader):
            for k in batch:
                batch[k] = batch[k].to(self.device)

            output = self.forward(**batch)
            probs = softmax(output.logits).detach().cpu()

            # get topk value and index
            topk_prob, topk_cate_idx = torch.topk(probs, topk)
            topk_cate = torchmapping(top_cate_idx, self.self.lbl2cate)
            
            # prerpare tuple list output
            batch_pred = [(c, p) for c, p in \
                    zip(topk_cate.reshape(-1), topk_prob.reshape(-1))]

            predictions += batch_pred

        return predictions