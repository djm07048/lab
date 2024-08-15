import os
import shutil
import random
import string


def generate_item_code():
    # Generate the first part 'E1'
    part1 = 'E1'

    # Generate the second part 'aaa' to 'fab'
    part2 = random.choice(['aaa', 'abb', 'bca', 'cad', 'fab', 'eac', 'ead', 'dda', 'aba', 'cca'])

    # Generate the third part from the given list
    part3 = random.choice(['ZG', 'PI', 'JS', 'HY', 'TH'])

    # Generate the fourth part '25'
    part4 = '25'

    # Generate the fifth part '0001' to '0030'
    part5 = f'{random.randint(1, 30):04}'

    return f'{part1}{part2}{part3}{part4}{part5}'


def create_copies(src_file, dest_folder, num_copies=20):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for _ in range(num_copies):
        item_code = generate_item_code()
        dest_file = os.path.join(dest_folder, f'{item_code}.pdf')
        shutil.copy(src_file, dest_file)


# Define source file and destination folder
src_file = 'item_folder/E1aaaHY250024.pdf'
dest_folder = 'item_folder'

# Create 20 copies with random item codes
create_copies(src_file, dest_folder)