from setuptools import setup, find_packages

setup(
    name='paddle-rifle',
    version='1.0rc',
    packages=["paddle_rifle"],
    url='https://github.com/GT-ZhangAcer/RIFLE_Module',
    license='MIT',
    author='zhanghongji',
    author_email='zhangacer@foxmail.com',
    description='本项目则为可用于PaddlePaddle的RIFLE优化策略封装版，支持普通API与高阶API，并且只需向训练代码中插入一行代码即可使用RIFLE策略。 '
)
