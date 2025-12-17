#!/bin/bash

coverage combine
coverage html
tar czf htmlcov.tar.gz htmlcov/

