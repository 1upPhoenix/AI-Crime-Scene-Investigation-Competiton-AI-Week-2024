import csv

def cipher_decryption(encrypted_msg, key):
    """
    Decrypts an encrypted message (in hex format) using XOR decryption with a repeating key.
    """
    # Convert hex message back to characters
    hex_to_uni = ""
    for i in range(0, len(encrypted_msg), 2):
        hex_to_uni += bytes.fromhex(encrypted_msg[i:i+2]).decode('utf-8')

    # XOR each character with the key to decrypt
    decrypt_text = ""
    key_itr = 0

    for i in range(len(hex_to_uni)):
        temp = ord(hex_to_uni[i]) ^ ord(key[key_itr])
        decrypt_text += chr(temp)
        key_itr += 1
        if key_itr >= len(key):
            key_itr = 0
    return decrypt_text


def decrypt_messages_from_csv(encrypted_file):
    """
    Reads a CSV file of encrypted messages and keys, decrypts each message,
    and writes only the decrypted results to a new CSV file.
    """
    with open(encrypted_file, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        decrypted_data = []

        for row in reader:
            encrypted_msg = row['encrypted_message']
            key = row['key']
            
            # Decrypt the message with the provided key
            decrypted_message = cipher_decryption(encrypted_msg, key)
            decrypted_data.append({
                'decrypted_message': decrypted_message
            })

    # Save only decrypted messages to a new CSV
    output_decrypted_path = r"C:\Users\ahmed\OneDrive\Desktop\AIQU\AI Crime Scene Investigation\challenge4\decrypted_messages.csv"
    with open(output_decrypted_path, mode='w', newline='') as csvfile:
        fieldnames = ['decrypted_message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decrypted_data)
    print(f"Decryption completed. Decrypted data saved to {output_decrypted_path}")


# Decrypt messages from the specified encrypted file
output_file_path = r"C:\Users\ahmed\OneDrive\Desktop\AIQU\AI Crime Scene Investigation\challenge4\encrypted_messages.csv"
decrypt_messages_from_csv(output_file_path)
