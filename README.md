# openai-learning

OpenAI の練習用リポジトリです。  
直接 OpenAI の API を使わずに langchain python を使っています。


# 起動方法

```bash
docker-compose up --build

# Database のリフレッシュを伴う起動
docker-compose --profile provision up
```

# Database のリフレッシュ

```bash
docker-compose run flyway-migrate
```

# 環境変数の設定

```bash
cp .env.sample .env
```

`.env` 中身を設定する

# その他

Dockerで動きますが、ホットリロード対応しているので、`app.py` を修正すれば反映されます。