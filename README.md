# AI Essayist

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


성균관대 AI X Bookathon 대회 4회차에 참여한 쿠봇팀의 모델 학습 코드를 관리하는 레포지토리 입니다.

## Quick Start

| TaskName | Description | Colab |
| --- | --- | --- |
| [Infer Samples](https://github.com/khu-bot/ai-essayist/blob/master/notebooks/infer_samples.ipynb) | 제목을 가지고 내용을 생성하는 수필 생성모델을 몇 개의 제목으로 내용을 생성해서 테스트합니다. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khu-bot/ai-essayist/blob/master/notebooks/infer_samples.ipynb) |
| [Infer Samples With Sum](https://github.com/khu-bot/ai-essayist/blob/master/notebooks/infer_samples_with_sum.ipynb) | 제목과 이전 내용의 요약문들을 가지고 다음 내용을 생성하는 수필 생성모델을 몇 개의 샘플로 내용을 생성해서 테스트합니다. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khu-bot/ai-essayist/blob/master/notebooks/infer_samples_with_sum.ipynb) |
| [Write Auto Essay](https://github.com/khu-bot/ai-essayist/blob/master/notebooks/write_auto_essay.ipynb) | 요약모델과 요약을 활용하는 수필 작성 모델을 모두 활용해 글 작성과 글 요약을 반복하며 특정 글자수가 될 때까지 글의 내용을 자동으로 생성합니다. 글 중간중간에 이상한 내용이 생성되곤 하는데 그러다가도 다시 관련된 내용이 생성되기 때문에 마지막에 사람이 보면서 이상한 부분을 없애줘야 합니다. (기본옵션으로 5000토큰을 생성하도록 되어있어 시간이 오래걸립니다. 빠르게 간단히 결과를 보려면 `num_generate_tokens`를 줄여주세요.) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khu-bot/ai-essayist/blob/master/notebooks/write_auto_essay.ipynb) |
| [Write Essay With Human](https://github.com/khu-bot/ai-essayist/blob/master/notebooks/write_essay_with_human.ipynb) | 요약모델과 요약을 활용하는 수필 작성 모델을 모두 활용해 글을 자동으로 작성합니다. 다만 모델이 글을 생성하면 사람이 생성한 문장들 중에서 어디까지를 실제로 사용할 지를 결정하면 그 선택된 텍스트만 내용으로 취급됩니다. 저희 팀이 대회에서 글을 생성한 방식입니다. seed를 고정해서 결과를 재현할 수 있습니다. (기본 옵션으로 `backup_indices`가 설정되어 있어 자동으로 문장을 선택해 글이 완성됩니다. 직접 사용할 텍스트를 선택하려면 **[Initial Condition]**에서 `backup_indices`를 빈 리스트로 초기화 해주세요.) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khu-bot/ai-essayist/blob/master/notebooks/write_essay_with_human.ipynb) |

- 모델 학습을 제외하고 저희가 모델로 했던 모든 작업은 각 Task마다 오른쪽에 있는 ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) 버튼으로 모두 실행해볼 수 있습니다. 이미 모든 값이 입력되어 있으므로 [Runtime]-[Run all]로 그냥 실행하셔도 결과를 볼 수 있습니다.

## Examples

### Train Model

```sh
python -m scripts.train \
    --model "EleutherAI/polyglot-ko-1.3b" \
    --dataset-dir dataset-v1/ \
    --batch-size 3 \
    --precision 16 \
    --strategy deepspeed \
    --accumulate-grad-batches 4 \
    --output-dir baseline-polyglot-1.3b \
    --disable-token-type-ids
```
- `--dataset-dir` 폴더에는 train.jsonl, dev.jsonl, test.jsonl파일이 존재해야합니다.
- json 데이터 하나의 형식은 [Datum](https://github.com/khu-bot/ai-essayist/blob/master/essayist/data.py#L8-L11)을 참고하세요.
