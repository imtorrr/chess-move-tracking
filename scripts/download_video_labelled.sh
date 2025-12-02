mkdir -p data/chess-pieces

gdown https://drive.google.com/uc?id=1PqmczMbOTtNqQL-wOYkA5YB5R6Cyndyi -O data/chess-pieces/video_labelled.tar.gz

tar -xvzf data/chess-pieces/video_labelled.tar.gz -C data/chess-pieces/
