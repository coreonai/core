#!/bin/bash
# Static server for coreontology.coreon.build (own docroot, port 8138).
# Routed via the DEDICATED cloudflared tunnel (llm-coreon.yml, bf329e41).
exec /home/paulyu/.pyenv/versions/3.9.12/bin/python3 -m http.server 8138 \
  --bind 127.0.0.1 --directory /raid/users/paul/workLLM/scratch-7b-sft/coreon_site
