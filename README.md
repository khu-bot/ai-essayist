# AI Essayist

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


성균관대 AI X Bookathon 대회 쿠봇팀의 모델 학습 코드를 관리하는 레포지토리 입니다.

## Examples

### kogpt2-base-v2

```sh
$ python -m scripts.train \
    --model "skt/kogpt2-base-v2" \
    --dataset-dir dataset-v1/ \
    --output-dir baseline-sk-kogpt2-v2 \
    --batch-size 8        
```

### kogpt2

```sh
python -m scripts.train \
    --model "skt/ko-gpt-trinity-1.2B-v0.5" \
    --dataset-dir dataset-v1/ \
    --batch-size 3 \
    --precision 16 \
    --strategy deepspeed \
    --accumulate-grad-batches 4 \
    --output-dir baseline-sk-trinity
```
- 트리니티의 경우 모델이 크기 때문에 FP16과 deepspeed를 모두 적용해야 16GB 이내의 메모리로 학습할 수 있습니다.

### Polyglot 1.3B

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
