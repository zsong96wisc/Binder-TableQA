# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors, The Google AI Language Team Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The WikiTableQuestions dataset is for the task of question answering on semi-structured HTML tables"""

import os
import datasets
from utils.wtq.utils import _load_table_w_page as _load_table

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{pasupat-liang-2015-compositional,
    title = "Compositional Semantic Parsing on Semi-Structured Tables",
    author = "Pasupat, Panupong  and
      Liang, Percy",
    booktitle = "Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = jul,
    year = "2015",
    address = "Beijing, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P15-1142",
    doi = "10.3115/v1/P15-1142",
    pages = "1470--1480",
}
"""

_DESCRIPTION = """\
Two important aspects of semantic parsing for question answering are the breadth of the knowledge source and the depth of
logical compositionality. While existing work trades off one aspect for another, this paper simultaneously makes progress 
on both fronts through a new task: answering complex questions on semi-structured tables using question-answer pairs as 
supervision. The central challenge arises from two compounding factors: the broader domain results in an open-ended set 
of relations, and the deeper compositionality results in a combinatorial explosion in the space of logical forms. We 
propose a logical-form driven parsing algorithm guided by strong typing constraints and show that it obtains significant
 improvements over natural baselines. For evaluation, we created a new dataset of 22,033 complex questions on Wikipedia
  tables, which is made publicly available.
"""

_HOMEPAGE = "https://ppasupat.github.io/WikiTableQuestions/"

_LICENSE = "CC-BY-SA-4.0 License"

_URL = "https://github.com/ppasupat/WikiTableQuestions/archive/refs/heads/master.zip"


class WikiTableQuestion(datasets.GeneratorBasedBuilder):
    """The WikiTableQuestions dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "table_id": datasets.Value("string"),
                    "table": {"page_title": datasets.Value("string"),
                              "header": datasets.features.Sequence(datasets.Value("string")),
                              "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))},
                    "answer_text": datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = os.path.join(dl_manager.download_and_extract(_URL), 'WikiTableQuestions-master')

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "data/random-split-1-dev_scalability_ablation.tsv"),
                            "data_dir": data_dir},
            ),
        ]

    def _generate_examples(self, filepath, data_dir):
        """Yields examples."""
        # data_id, question, table_id, gold_result_str
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # skip the header
                if idx == 0:
                    continue
                data_id, question, table_id, gold_result_str = line.strip("\n").split("\t")
                if 'rows' in data_id:
                    continue
                table_id = table_id.replace('csv', 'expanded_csv_ori_100_200_500', 1)
                table_path = os.path.join(data_dir, table_id.replace('.csv', '.tsv'))
                page_title_path = os.path.join(
                    data_dir, 'page/',
                    '/'.join(table_id.split('/')[1:]).replace("csv", "page").replace(".tsv", ".json")
                )

                gold_result = gold_result_str.split('|')
                yield idx, {
                    "id": data_id,
                    "question": question,
                    "table_id": table_id,
                    "table": _load_table(
                        table_path=table_path,
                        page_title_path=page_title_path
                    ),
                    # convert the .csv postfix to .tsv, for easier read-in
                    "answer_text": gold_result,
                }
