from setuptools import find_packages, setup

setup(
    name="ai-essayist",
    version="1.0.0",
    description="인공지능 북커톤 대회 모델 학습을 위한 레포지토리",
    python_requires=">=3.7",
    install_requires=[],
    url="https://github.com/khu-bot/ai-essayist.git",
    packages=find_packages(exclude=["tests"]),
)
