import csv

def cipher_encryption(message, key):
    """
    Encrypts a message using XOR encryption with a repeating key.
    """
    encrypt_hex = ""
    key_itr = 0

    for i in range(len(message)):
        temp = ord(message[i]) ^ ord(key[key_itr])
        encrypt_hex += hex(temp)[2:].zfill(2)
        key_itr += 1
        if key_itr >= len(key):
            key_itr = 0
    return encrypt_hex


def encrypt_messages_to_csv(input_file, output_file):
    """
    Reads messages and keys from an input CSV, encrypts each message,
    and writes the results to an output CSV.
    """
    with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Ensure headers are lowercase and stripped of extra spaces
        reader.fieldnames = [header.strip().lower() for header in reader.fieldnames]
        
        encrypted_data = []

        for row in reader:
            # Use stripped headers to access the correct columns
            message = row['message'].strip()
            key = row['key'].strip()
            encrypted_message = cipher_encryption(message, key)
            
            # Store original message, key, and encrypted message in the output
            encrypted_data.append({
                'encrypted_message': encrypted_message,
                'key': key  # Save the key for decryption
            })

    # Write encrypted data to the output CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['encrypted_message', 'key']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(encrypted_data)
    print(f"Encryption completed. Encrypted data saved to {output_file}")


# Encrypt messages from the specified file
input_file_path = r"C:\Users\ahmed\OneDrive\Desktop\AIQU\AI Crime Scene Investigation\challenge4\plain_messages.csv"
output_file_path = r"C:\Users\ahmed\OneDrive\Desktop\AIQU\AI Crime Scene Investigation\challenge4\encrypted_messages.csv"

encrypt_messages_to_csv(input_file_path, output_file_path)
