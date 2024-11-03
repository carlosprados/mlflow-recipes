#!/bin/bash

sphinx-apidoc -o . ../mlflowx

make html

caddy file-server --listen :8000
