FROM redis:7-alpine

COPY redis.conf /usr/local/etc/redis/redis.conf

# Start with custom config
CMD [ "redis-server", "/usr/local/etc/redis/redis.conf" ]
