import logging
import os
import sys

# Add the parent directory to the sys.path to import the local module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parsers import FileParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # List of file names to test
    filenames = ["obama.txt", "obama.pdf", "obama-ocr.pdf"]

    # The files are in a folder named "sources" and this file is in the tests folder
    # so we need to go up one level to access the "sources" folder
    files = [f"sources/{filename}" for filename in filenames]

    # Loop through the files and use the FileParser to parse each one
    for filename in files:
        try:
            # Create a FileParser instance with the filename
            parser = FileParser(filepath=filename)
            # Parse the file and print the output
            content = parser.parse()
            print(f"Content of {filename}:")
            print(content[:500])  # Print the first 500 characters to avoid too much output
            print("--------------------------------------------------")
        except Exception as e:
            # Print any errors encountered during parsing
            logging.error(f"Failed to process file '{filename}': {e}")


if __name__ == "__main__":
    main()
