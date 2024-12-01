# Karpathy GPT-2

This repository contains the implementation of Karpathy's GPT-2 model.

This repository includes a Docker Compose setup that allows for different ways of running scripts and visualizing Docker on the host machine.

## Resources

- [Karpathy's Video](https://www.youtube.com/watch?v=l8pRSuU81PU&t=5294s)
- [Karpathy's repo](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbDJnRlhLTVNFM0E3cGpkczBWM1pLZkkwcU5lQXxBQ3Jtc0tsQXFwMHZsdlBpOHZSOWxQRmt3dEZEeFhEb3c3OVlUdTdBZ1NhZFlYVW1DOThtTm1SbV9YbnVDeGRFaFpxMWpCaVBVLWI2RWVONVVuRnI2aG9QT3pJc25oUVhKMkR4VjFWZVlSMERNWGV5ZElpYk9Xbw&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fbuild-nanogpt&v=l8pRSuU81PU)
- [Attention is All You Need paper](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbldFWkxOa2hwM3ZUU0N4ZHhVQ1FnZEgxVFcxZ3xBQ3Jtc0tsai1MT1daNG0xTm03dnJoeHkzZ1NUMEttUm5CdHdvQmZlMko0R2VsTHh6WGhzUGZucGZ1aFN1Y3M2d1hLajRvWW9DNnVVZnFKalF6MERsWVVyUjYyZUlFT2JYUjdGVV8xT1NOUUZoVDd3cHlNcF9BWQ&q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762&v=l8pRSuU81PU)
- [OpenAI GPT-3 paper](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa1ZFZTI0M1BuWGZjeGhjb0tuQzF2cVVScHFHZ3xBQ3Jtc0tuMlJHTjI2MWhZQ25zU051TjVXenJuLWRGd056bWJEYlhBUVpoMklIQWtEaWlkRXhQdm5TNHh6R0pRY1hxX2pSWGFIVkdYc0hlMkRMT0NxcWotMGdoeFVySTVyNFBVU25wUGdPTHBvY0JKc0lZdFRrNA&q=https%3A%2F%2Farxiv.org%2Fabs%2F2005.14165&v=l8pRSuU81PU)
- [OpenAI GPT-2 paper](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGlWX2Y1NnpRQmpKQmxJUFVjVnJHNUdnVmhrd3xBQ3Jtc0trY19xZi02NGJOVU9mN3dwdTloVnNFMjNzSWd3VjVUaTFySXUwMlgxM2NHSDZ5S2tCWnBqenBzV2FXSXdVZHhLREFTVTZKTWRhZU1EWWhXZ19YT0JrX1JmZWxyam9QVzBpS2hHanRnRG5ROTFFM2dBUQ&q=https%3A%2F%2Fd4mucfpksywv.cloudfront.net%2Fbetter-language-models%2Flanguage_models_are_unsupervised_multitask_learners.pdf-&v=l8pRSuU81PU)

## Building the Docker Image

To build the Docker image, run the following command:
```sh
docker-compose up --build -d
```

To access the running container, use:
```sh
docker exec -it gpt-2 /bin/bash
```

## Visualization

### 1. Jupyter Notebook

To run Jupyter Notebook in Docker, use the container ID:
```sh
jupyter notebook --ip=172.28.0.10 --allow-root
```

On your local machine, replace the resulting URL with `localhost`. For example:
```
http://172.18.0.2:8888/tree?token=e8b6df7a2a -> http://localhost:8888/tree?token=e8b6df7a2a
```

### 2. XServer

Install an XServer on your Windows host, such as VcXsrv. Use `play_xserver.py` as an example.

### 3. Matplotlib Web Server

Map the desired port in `docker-compose`. Use `play_web.py` as an example.