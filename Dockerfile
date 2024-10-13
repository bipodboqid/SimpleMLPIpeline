# ベースイメージとしてPython 3.10を使用
FROM python:3.10

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストールするrequirements.txtをコピー
COPY Ph05_ToCloudRun/requirements.txt /app/

# パッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# test.py スクリプトと test.jpeg をコピー
COPY Ph05_ToCloudRun/test.py /app/
COPY Ph05_ToCloudRun/mnist_156.jpg /app/

EXPOSE 8080

# Cloud Buildによるデプロイで実行するためのエントリーポイント
ENTRYPOINT ["python", "/app/test.py"]