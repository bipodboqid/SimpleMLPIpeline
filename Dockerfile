# ベースイメージとしてPython 3.10を使用
FROM python:3.10

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストールするrequirements.txtをコピー
COPY Ph05_ToCloudRun/requirements.txt .

# パッケージをインストール
RUN pip install --no-cache-dir -r Ph05_ToCloudRun/requirements.txt

# test.py スクリプトと test.jpeg をコピー
COPY Ph05_ToCloudRun/test.py .
COPY Ph05_ToCloudRun/mnist_156.jpg .

# Cloud Buildによるデプロイで実行するためのエントリーポイント
ENTRYPOINT ["python", "Ph05_ToCloudRun/test.py"]