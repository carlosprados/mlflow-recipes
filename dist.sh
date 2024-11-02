#!/bin/bash

# Script to manage the lifecycle of a Python package publication on PyPI.
# The script allows creating distributions and uploading to PyPI separately or together.

# Set default values
distribute=false
upload=false

display_help() {
  echo "Usage: $0 [option...]" >&2
  echo
  echo "   -d, --distribute          Create package distributions"
  echo "   -u, --upload              Upload package to PyPI"
  echo "   -h, --help                Show help"
  echo
  exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -d|--distribute)
      distribute=true
      shift
      ;;
    -u|--upload)
      upload=true
      shift
      ;;
    -h|--help)
      display_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      display_help
      exit 1
      ;;
  esac
done

# Ensure at least one action is chosen
if [ "$distribute" = false ] && [ "$upload" = false ]; then
  echo "Error: No action specified. Use -d to create distributions or -u to upload." >&2
  display_help
  exit 1
fi

# Create distributions if requested
if [ "$distribute" = true ]; then
  echo "Creating distribution archives..."
  python setup.py sdist bdist_wheel
  if [ $? -ne 0 ]; then
    echo "Error: Distribution creation failed." >&2
    exit 1
  fi
  echo "Distribution archives created successfully."
fi

# Upload to PyPI if requested
if [ "$upload" = true ]; then
  echo "Uploading distributions to PyPI..."
  twine upload dist/*
  if [ $? -ne 0 ]; then
    echo "Error: Upload to PyPI failed." >&2
    exit 1
  fi
  echo "Upload to PyPI completed successfully."
fi

exit 0
