#!/bin/bash

if [ ! -d "venv" ]; then
	echo "creating virtual environment..."
	python -m venv venv
	echo "done."
fi

if [ -f "requirements.txt" ]; then
	if [ -z "$VIRTUAL_ENV" ]; then
		echo "activating venv..."
		source venv/bin/activate
		echo "venv activated."
	fi

	echo "installing dependencies from requirements.txt..."
	pip install -r requirements.txt
	echo "done."
else
	echo "requirements.txt not found." >&2
fi
