
# Spring installation

Installation of Spring on our cluster was a hassle due to the old dependency on Transformers v2 which in turn relies on tokenizers v0.7.0.
This was not easy to install because it doesn't come with prebuilt wheels and relies on an old, nightly Rust build. So to solve this
I built the wheel on our server, and then moved that wheel to the cluster, and installed it there. To do so:

```shell
rustup default nightly-2020-03-12

git clone https://github.com/huggingface/tokenizers.git
cd tokenizers
git checkout tags/python-v0.7.0
python -m pip install setuptools_rust
python -m pip wheel .
```
