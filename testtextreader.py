# Define the function to read and print the file content
def read_and_print_file(filename):
    # Open the file
    with open(filename, 'r') as file:
        # Read the content
        data = file.read()

    # Print the content
    print(data)

# Call the function with your file
read_and_print_file('testdoc.txt')
